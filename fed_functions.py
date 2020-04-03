from utils import *
import pickle
import copy
import sys
import glob
import os
import shutil
import random
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Lock, Manager

from matching.pfnm import layer_wise_group_descent
from matching.pfnm import block_patching, patch_weights
from matching_performance import compute_full_cnn_accuracy
from local_train_functions import local_retrain, local_retrain_fedavg, local_retrain_fedprox, local_retrain_dummy, reconstruct_local_net

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, shortest_path
import numpy as np
from collections import defaultdict

args_datadir = "./data/cifar10"
# torch.multiprocessing.set_sharing_strategy('file_system')

def oneshot_matching(nets_list, model_meta_data, layer_type, net_dataidx_map, 
                            averaging_weights, args, 
                            device="cpu"):
    # starting the neural matching
    models = nets_list
    cls_freqs = traindata_cls_counts
    n_classes = args.n_class
    it=5
    sigma=args_pdm_sig 
    sigma0=args_pdm_sig0
    gamma=args_pdm_gamma
    assignments_list = []
    
    batch_weights = pdm_prepare_full_weights_cnn(models, device=device)
    raw_batch_weights = copy.deepcopy(batch_weights)
    
    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gamma = 7.0
    sigma = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None

    for layer_index in range(1, n_layers):
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
             batch_weights=batch_weights, 
             layer_index=layer_index,
             sigma0_layers=sigma0, 
             sigma_layers=sigma, 
             batch_frequencies=batch_freqs, 
             it=it, 
             gamma_layers=gamma, 
             model_meta_data=model_meta_data,
             model_layer_type=layer_type,
             n_layers=n_layers,
             matching_shapes=matching_shapes,
             args=args
             )
        assignments_list.append(assignment)
        
        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and ('conv' in type_of_prev_layer or 'features' in type_of_this_layer))
        
        if first_fc_identifier:
            first_fc_index = layer_index
        
        matching_shapes.append(L_next)
        #tempt_weights = [batch_weights[0][i] for i in range(2 * layer_index - 2)] + [copy.deepcopy(layer_hungarian_weights) for _ in range(num_workers)]
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model)
                elif l_type == "fc":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next, assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])

        retrained_nets = []
        for worker_index in range(num_workers):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512, dataidxs)
            logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index, 2 * (layer_index + 1) - 2))
            retrained_cnn = local_retrain_dummy((train_dl_local,test_dl_local), tempt_weights[worker_index], args, 
                                            freezing_index=(2 * (layer_index + 1) - 2), device=device)
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
        pickle.dump(batch_weights, weights_file)

    last_layer_weights_collector = []

    for i in range(num_workers):
        # firstly we combine last layer's weight and bias
        bias_shape = batch_weights[i][-1].shape
        last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
        last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
        # the directed normalization doesn't work well, let's try weighted averaging
        last_layer_weights_collector.append(last_layer_weights)

    last_layer_weights_collector = np.array(last_layer_weights_collector)
    
    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

    for i in range(n_classes):
        avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
        for j in range(num_workers):
            avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
        avg_last_layer_weight[:, i] = avg_weight_collector

    #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    for i in range(num_layers):
        if i < (num_layers - 2):
            matched_weights.append(batch_weights[0][i])

    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])
    return matched_weights, assignments_list

def get_retrained_net(args, layer_index, worker_index, dataidxs, tmp_weight, device, d, lock):

    train_dl_local, test_dl_local = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512, dataidxs)
    logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index, 2 * (layer_index + 1) - 2))
    retrained_cnn = local_retrain((train_dl_local,test_dl_local), tmp_weight, args, 
                                    freezing_index=(2 * (layer_index + 1) - 2), device=device)
    with lock:
        d[worker_index]=retrained_cnn.cpu()

def BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map, 
                            averaging_weights, args, 
                            device="cpu"):
    # starting the neural matching
    models = nets_list
    cls_freqs = args.traindata_cls_counts
    n_classes = args.n_class
    it=5
    sigma=args.args_pdm_sig 
    sigma0=args.args_pdm_sig0
    gamma=args.args_pdm_gamma
    assignments_list = []
    
    batch_weights = pdm_prepare_full_weights_cnn(models, device=device)
    raw_batch_weights = copy.deepcopy(batch_weights)
    
    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gamma = 7.0
    sigma = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None

    for layer_index in range(1, n_layers):
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
             batch_weights=batch_weights, 
             layer_index=layer_index,
             sigma0_layers=sigma0, 
             sigma_layers=sigma, 
             batch_frequencies=batch_freqs, 
             it=it, 
             gamma_layers=gamma, 
             model_meta_data=model_meta_data,
             model_layer_type=layer_type,
             n_layers=n_layers,
             matching_shapes=matching_shapes,
             args=args
             )
        assignments_list.append(assignment)
        
        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and ('conv' in type_of_prev_layer or 'features' in type_of_this_layer))
        
        if first_fc_identifier:
            first_fc_index = layer_index
        
        matching_shapes.append(L_next)
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model)
                elif l_type == "fc":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next, assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])


        if args.multiprocess:
            device_list = ['cuda:%d'%(i%args.gpu_num) for i in range(num_workers)]
            mp.set_start_method('spawn', force=True)
            m = Manager()
            d = m.dict()
            lock = Lock()


            processes = []
            for rank in range(num_workers):
                p = mp.Process(target=get_retrained_net, args=(args, layer_index, rank, net_dataidx_map[rank], tempt_weights[rank], device_list[rank], d, lock))
                # We first train the model across `num_processes` processes
                p.start()
                processes.append(p)
                if len(processes)>=16: # restrict maximum processes
                    for p in processes:
                        p.join()
                    processes = []
            for p in processes:
                p.join()
            retrained_nets = [d[rank] for rank in range(num_workers)]
        
        else:
            retrained_nets = []
            for worker_index in range(num_workers):
                dataidxs = net_dataidx_map[worker_index]
                train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs)

                logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index, 2 * (layer_index + 1) - 2))
                retrained_cnn = local_retrain((train_dl_local,test_dl_local), tempt_weights[worker_index], args, 
                                                freezing_index=(2 * (layer_index + 1) - 2), device=device)
                retrained_nets.append(retrained_cnn)
        
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
        pickle.dump(batch_weights, weights_file)

    last_layer_weights_collector = []

    for i in range(num_workers):
        # firstly we combine last layer's weight and bias
        bias_shape = batch_weights[i][-1].shape
        last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
        last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
        # the directed normalization doesn't work well, let's try weighted averaging
        last_layer_weights_collector.append(last_layer_weights)

    last_layer_weights_collector = np.array(last_layer_weights_collector)
    
    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

    for i in range(n_classes):
        avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
        for j in range(num_workers):
            avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
        avg_last_layer_weight[:, i] = avg_weight_collector

    #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    for i in range(num_layers):
        if i < (num_layers - 2):
            matched_weights.append(batch_weights[0][i])

    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])
    return matched_weights, assignments_list

""" for multiprocessing """
def get_retrained_fed_net(args, worker_index, dataidxs, tmp_weight, device, d, lock, assignments_list=None):
    train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 512, dataidxs)
    if args.comm_type=='fedavg' or 'fedmf' in args.comm_type:
        retrained_cnn = local_retrain_fedavg((train_dl_local,test_dl_local), tmp_weight, args, device=device)
    elif args.comm_type=='fedprox':
        retrained_cnn = local_retrain_fedprox((train_dl_local,test_dl_local), tmp_weight, args, device=device)
    elif args.comm_type=='fedma':
        recons_local_net = reconstruct_local_net(tmp_weight, args, ori_assignments=assignments_list, worker_index=worker_index)
        retrained_cnn = local_retrain((train_dl_local,test_dl_local), recons_local_net, args,
                                mode="bottom-up", freezing_index=0, ori_assignments=None, device=device)
    with lock:
        d[worker_index]=retrained_cnn.cpu()


def fed_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
             averaging_weights, args, train_dl_global, test_dl_global,
             comm_round=2, device="cpu", assignments_list=None):


    logging.info("=="*15)

    for cr in range(comm_round):

        if cr<10:
            args.comm_type='fedavg'
            args.retrain_epochs=20
        else:
            args.comm_type='fedmfv3'
            args.retrain_epochs=20
        
        logger.info("Communication round : {}".format(cr))
        logger.info(args.comm_type)

        if args.multiprocess:
            workers = random.sample(range(args.n_nets), args.clients_per_round)
            num_workers = args.clients_per_round

            device_list = ['cuda:%d'%(i%args.gpu_num) for i in range(num_workers)]
            mp.set_start_method('spawn', force=True)
            m = Manager()
            d = m.dict()
            lock = Lock()

            processes = []
            for rank in range(num_workers):
                p = mp.Process(target=get_retrained_fed_net, args=(args, rank, net_dataidx_map[rank], batch_weights[rank], device_list[rank], d, lock, assignments_list))
                p.start()
                processes.append(p)
                if len(processes)>=16: # restrict maximum processes
                    for p in processes:
                        p.join()
                    processes = []
            for p in processes:
                p.join()
            worker_list = [rank for rank in d]
            print(worker_list)
            worker_list = sorted(worker_list)
            print(worker_list)
            retrained_nets = [d[rank] for rank in worker_list]

        else:
            retrained_nets = []
            for worker_index in random.sample(range(args.n_nets), args.clients_per_round):
                dataidxs = net_dataidx_map[worker_index]
                train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 512, dataidxs)
                
                # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
                if args.comm_type=='fedavg' or 'fedmf' in args.comm_type:
                    retrained_cnn = local_retrain_fedavg((train_dl_local,test_dl_local), batch_weights[worker_index], args, device=device)
                elif args.comm_type=='fedprox':
                    retrained_cnn = local_retrain_fedprox((train_dl_local,test_dl_local), batch_weights[worker_index], mu=0.001, args=args, device=device)
                elif args.comm_type=='fedma':
                    recons_local_net = reconstruct_local_net(batch_weights[worker_index], args, ori_assignments=assignments_list, worker_index=worker_index)
                    retrained_cnn = local_retrain((train_dl_local,test_dl_local), recons_local_net, args,
                                            mode="bottom-up", freezing_index=0, ori_assignments=None, device=device)

                retrained_nets.append(retrained_cnn)

        if args.comm_type=='fedma':
            hungarian_weights, assignments_list = BBP_MAP(retrained_nets, model_meta_data, layer_type, net_dataidx_map, averaging_weights, args, device=device)
            logger.info("After retraining and rematching for comm. round: {}, we measure the accuracy ...".format(cr))
            averaged_weights = hungarian_weights

        elif args.comm_type=='fedmf': # maximum flow 0/1
            batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)
            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
            averaged_weights = []
            previous_kernel_order = defaultdict(list) # keep kernel order with each worker, worker index --> kernel order list
            num_layers = len(batch_weights[0])
            num_workers = len(batch_weights)
            thres = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

            for layer_idx in range(num_layers):
                if layer_idx<=10 and layer_idx%2==0:
                    # t0 = time.time()
                    num_kernels = batch_weights[0][layer_idx].shape[0] # dimension: (c_out, c_in*k*k)
                    capacity_matrix = [[0 for k in range(2*num_workers*num_kernels+2)] for m in range(2*num_workers*num_kernels+2)] # +2 for source and sink
                    order_list = [i for i in range(num_workers)]
                    group_list = [(order_list[i], order_list[i+1]) for i in range(len(order_list)-1)]
                    
                    ''' implementation of node capacity == nodes split and connect with one edge '''
                    ''' guarantee one flow in and one flow out '''
                    for worker in order_list:
                        for kernel_idx in range(num_kernels):
                            capacity_matrix[num_kernels*worker*2+kernel_idx][num_kernels*(worker*2+1)+kernel_idx]=1


                    ''' permutate kernel dimension with previous order '''
                    if layer_idx != 0:
                        for worker_idx in range(num_workers):
                            fix_kernel_list = []
                            for kernel_idx in range(num_kernels):
                                orig_kernel = batch_weights[worker_idx][layer_idx][kernel_idx]
                                orig_kernel = orig_kernel.reshape(orig_kernel.shape[0]//9, 3, 3)
                                fix_kernel = np.concatenate([orig_kernel[k:k+1] for k in previous_kernel_order[worker_idx]])
                                fix_kernel = fix_kernel.reshape(1, orig_kernel.shape[0]*9)
                                fix_kernel_list.append(fix_kernel)
                            batch_weights[worker_idx][layer_idx] = np.concatenate(fix_kernel_list)


                    ''' decide which two kernels can be connected '''
                    connections = 0
                    value_count = [0, 0, 0, 0, 0, 0, 0]
                    for (worker1, worker2) in group_list:
                    # for worker1 in range(num_workers):
                    #     for worker2 in range(0, worker1):
                        worker1_end = worker1*2+1
                        worker2_start = worker2*2
                        for kernel1_idx in range(num_kernels):
                            for kernel2_idx in range(num_kernels):
                                distance = l2_norm(batch_weights[worker1][layer_idx][kernel1_idx], batch_weights[worker2][layer_idx][kernel2_idx])
                                # if is_match(batch_weights[worker1][layer_idx][kernel1_idx], batch_weights[worker2][layer_idx][kernel2_idx]):
                                if distance < thres[layer_idx//2]:
                                    capacity_matrix[num_kernels*worker1_end+kernel1_idx][num_kernels*worker2_start+kernel2_idx]=1
                                    connections += 1
                                for i in range(7):
                                    if distance>1:
                                        value_count[i]+=1
                                        break
                                    distance = distance*10
                    logging.info(value_count)
                    
                    # t1 = time.time()
                    first_worker = order_list[0]*2
                    last_worker = order_list[-1]*2+1
                    source = 2*num_workers*num_kernels
                    sink = 2*num_workers*num_kernels+1
                    for kernel_idx in range(num_kernels):
                        capacity_matrix[source][num_kernels*first_worker+kernel_idx]=1 # source to first worker
                        capacity_matrix[num_kernels*last_worker+kernel_idx][sink]=1 # last worker to sink
                    
                    # t2 = time.time()
                    capacity_graph = csr_matrix(capacity_matrix)
                    flow = maximum_flow(capacity_graph, source, sink)
                    residual_graph = flow.residual.toarray()
                    value = flow.flow_value
                    logging.info("round: {}, layer: {}, connections: {}, maximum_match: {}, ratio: {}".format(cr, layer_idx, connections, value, value/num_kernels))
                    # dist_matrix, predecessors = shortest_path(csgraph=capacity_graph, directed=False, indices=0, return_predecessors=True)
                    
                    # t3 = time.time()
                    first_worker = order_list[0]*2
                    kernel_order = [i for i in range(num_kernels)]
                    kernel_order = sorted(kernel_order, key=lambda x: -residual_graph[source][num_kernels*first_worker+x])
                    batch_weights[first_worker][layer_idx] = np.concatenate([batch_weights[first_worker][layer_idx][k:k+1] for k in kernel_order])
                    previous_kernel_order[order_list[0]] = kernel_order

                    for (worker1, worker2) in group_list:
                        worker1_end = worker1*2+1
                        worker2_start = worker2*2

                        partial_graph = residual_graph[num_kernels*worker1_end:num_kernels*(worker1_end+1), num_kernels*worker2_start:num_kernels*(worker2_start+1)]
                        matching_dict = defaultdict(lambda:-1)
                        new_kernel_order = []
                        check_for_no_matching=False
                        for kernel_idx in kernel_order:
                            if check_for_no_matching:
                                assert(np.max(partial_graph[kernel_idx])==0) # check for no other matching
                            else:
                                if np.max(partial_graph[kernel_idx])==1:
                                    matching_dict[np.argmax(partial_graph[kernel_idx])] = kernel_idx
                                    new_kernel_order.append(np.argmax(partial_graph[kernel_idx]))
                                else:
                                    check_for_no_matching=True
                        for kernel_idx in range(num_kernels):
                            if matching_dict[kernel_idx]==-1:
                                new_kernel_order.append(kernel_idx)
                        
                        # print(new_kernel_order)
                        # print(num_kernels)
                        assert len(new_kernel_order)==num_kernels
                        kernel_order = new_kernel_order
                        batch_weights[worker2][layer_idx] = np.concatenate([batch_weights[worker2][layer_idx][k:k+1] for k in kernel_order])
                        previous_kernel_order[worker2] = kernel_order
                    # t4 = time.time()
                
                elif layer_idx<=11 and layer_idx%2==1:
                    ''' permutate bias according to previous order '''
                    for worker_idx in range(num_workers):
                        bias = batch_weights[worker_idx][layer_idx]
                        batch_weights[worker_idx][layer_idx] = np.concatenate([bias[i:i+1] for i in previous_kernel_order[worker_idx]])

                elif layer_idx==12: # first fc
                    ''' permutate bias according to previous order '''
                    for worker_idx in range(num_workers):
                        fc = batch_weights[worker_idx][layer_idx]
                        fc = fc.reshape(256, 16, 512)
                        fc = np.concatenate([fc[i:i+1] for i in previous_kernel_order[worker_idx]])
                        fc = fc.reshape(4096, 512)
                        batch_weights[worker_idx][layer_idx] = fc

                avegerated_weight = sum([b[layer_idx] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
                averaged_weights.append(avegerated_weight)
                # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(t1-t0, t2-t1, t3-t2, t4-t3))
        
        elif args.comm_type=='fedmfv2': # maximum flow sum
            batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)
            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
            averaged_weights = []
            # previous_kernel_order = defaultdict(list) # keep kernel order with each worker, worker index --> kernel order list
            num_layers = len(batch_weights[0])
            num_workers = len(batch_weights)

            for layer_idx in range(num_layers):
                if layer_idx<=10 and layer_idx%2==0:
                    # t0 = time.time()
                    num_kernels = batch_weights[0][layer_idx].shape[0] # dimension: (c_out, c_in*k*k)
                    capacity_matrix = [[0 for k in range(num_workers*num_kernels+2)] for m in range(num_workers*num_kernels+2)] # +2 for source and sink
                    order_list = [i for i in range(num_workers)]
                    group_list = [(order_list[i], order_list[i+1]) for i in range(len(order_list)-1)]


                    ''' permutate kernel dimension with previous order '''
                    if layer_idx != 0:
                        for worker_idx in range(num_workers):
                            fix_kernel_list = []
                            for kernel_idx in range(num_kernels):
                                orig_kernel = batch_weights[worker_idx][layer_idx][kernel_idx]
                                orig_kernel = orig_kernel.reshape(orig_kernel.shape[0]//9, 3, 3)
                                fix_kernel = np.concatenate([orig_kernel[k:k+1] for k in all_kernel_order[worker_idx]])
                                fix_kernel = fix_kernel.reshape(1, orig_kernel.shape[0]*9)
                                fix_kernel_list.append(fix_kernel)
                            batch_weights[worker_idx][layer_idx] = np.concatenate(fix_kernel_list)


                    ''' assign capacity '''
                    connections = 0
                    for (worker1, worker2) in group_list:
                        for kernel1_idx in range(num_kernels):
                            for kernel2_idx in range(num_kernels):
                                capacity_matrix[num_kernels*worker1+kernel1_idx][num_kernels*worker2+kernel2_idx] =\
                                similarity(batch_weights[worker1][layer_idx][kernel1_idx], batch_weights[worker2][layer_idx][kernel2_idx])
                    
                    # t1 = time.time()
                    first_worker = order_list[0]
                    last_worker = order_list[-1]
                    source = num_workers*num_kernels
                    sink = num_workers*num_kernels+1
                    for kernel_idx in range(num_kernels):
                        capacity_matrix[source][num_kernels*first_worker+kernel_idx]=9999 # source to first worker
                        capacity_matrix[num_kernels*last_worker+kernel_idx][sink]=9999 # last worker to sink
                    
                    # t2 = time.time()
                    all_kernel_order = [[] for i in range(num_workers)]

                    def search_path(graph, source, sink, best_path, best_path_value):
                        ''' trace dominate flow: dfs '''
                        path_candidate = []
                        for i in range(num_kernels*num_workers+2):
                            if graph[source][i]>0:
                                if i==sink:
                                    path_candidate.append([[], graph[source][i]])
                                    continue

                                if best_path_value[i]==-1:
                                    best_path, best_path_value = search_path(graph, i, sink, best_path, best_path_value)

                                if best_path_value[i]>0:
                                    path_candidate.append([[node for node in best_path[i]], best_path_value[i]+graph[source][i]])
                                
                                
                                # if len(path_candidate)==0 or res[1]>path_candidate[0][1]:
                                #     path_candidate = [[res[0], res[1]]]

                        # print(path_candidate)
                        if len(path_candidate)==0:
                            best_path[source] = [source]
                            best_path_value[source] = 0
                        else:
                            path_candidate = sorted(path_candidate, key=lambda x: x[1])
                            path_candidate[-1][0].append(source)
                            best_path[source] = path_candidate[-1][0]
                            best_path_value[source] = path_candidate[-1][1]
                        return best_path, best_path_value


                    

                    while True:
                        capacity_graph = csr_matrix(capacity_matrix)
                        flow = maximum_flow(capacity_graph, source, sink)
                        residual_graph = flow.residual.toarray()
                        value = flow.flow_value
                        logger.info("round: {}, layer: {}, flow_value: {}".format(cr, layer_idx, value))
                        if value==0:
                            break

                        while True:
                            t1 = time.time()
                            best_path = defaultdict(list)
                            best_path_value = defaultdict(lambda: -1)
                            best_path, best_path_value = search_path(residual_graph, source, sink, best_path, best_path_value)
                            if best_path_value[source]<=0:
                                break
                            path = best_path[source][:len(best_path[source])-1][::-1]
                            t2 = time.time()
                            logger.info("match: {}, dominate_flow: {}, time: {:.3f}".format(path, best_path_value[source], t2-t1))
                            for wid, kid in enumerate(path):
                                all_kernel_order[wid].append(kid%num_kernels)
                                capacity_matrix[kid] = [0 for i in range(num_workers*num_kernels+2)]
                                residual_graph[kid] = 0
                            
                            
                        

                    # previous_max_flow = defaultdict(lambda: -1)
                    # path, dominate_flow, previous_max_flow = search_path(residual_graph, previous_max_flow, source, sink, 999)
                    # while dominate_flow>0:
                    #     t1 = time.time()
                        
                    #     path = path[::-1]
                    #     for wid, kid in enumerate(path):
                    #         all_kernel_order[wid].append(kid%num_kernels)
                    #         residual_graph[kid] = 0
                        
                    #     t2 = time.time()

                    #     previous_max_flow = defaultdict(lambda: -1)
                    #     path, dominate_flow, previous_max_flow = search_path(residual_graph, previous_max_flow, source, sink, 999)

                    #     t3 = time.time()
                    #     logger.info("match: {}, dominate_flow: {}, time: {:.3f}, {:.3f}".format(path, dominate_flow, t2-t1, t3-t2))

                    # t3 = time.time()

                    for worker_idx in order_list:
                        for kernel_idx in range(num_kernels):
                            if kernel_idx not in all_kernel_order[worker_idx]:
                                all_kernel_order[worker_idx].append(kernel_idx)
                        assert len(all_kernel_order[worker_idx])==num_kernels
                        batch_weights[worker_idx][layer_idx] = np.concatenate([batch_weights[worker_idx][layer_idx][k:k+1] for k in all_kernel_order[worker_idx]])

                    # t4 = time.time()

                elif layer_idx<=11 and layer_idx%2==1:
                    ''' permutate bias according to previous order '''
                    for worker_idx in range(num_workers):
                        bias = batch_weights[worker_idx][layer_idx]
                        batch_weights[worker_idx][layer_idx] = np.concatenate([bias[i:i+1] for i in all_kernel_order[worker_idx]])

                elif layer_idx==12: # first fc
                    ''' permutate bias according to previous order '''
                    for worker_idx in range(num_workers):
                        fc = batch_weights[worker_idx][layer_idx]
                        fc = fc.reshape(256, 16, 512)
                        fc = np.concatenate([fc[i:i+1] for i in all_kernel_order[worker_idx]])
                        fc = fc.reshape(4096, 512)
                        batch_weights[worker_idx][layer_idx] = fc

                
                avegerated_weight = sum([b[layer_idx] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
                averaged_weights.append(avegerated_weight)
                # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(t1-t0, t2-t1, t3-t2, t4-t3))

        elif args.comm_type=='fedmfv3': # maximum flow bottleneck
            batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)
            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
            averaged_weights = []
            # previous_kernel_order = defaultdict(list) # keep kernel order with each worker, worker index --> kernel order list
            num_layers = len(batch_weights[0])
            num_workers = len(batch_weights)

            for layer_idx in range(num_layers):
                if layer_idx<=10 and layer_idx%2==0:
                    # t0 = time.time()
                    num_kernels = batch_weights[0][layer_idx].shape[0] # dimension: (c_out, c_in*k*k)
                    capacity_matrix_group = [[[0 for k in range(num_workers*num_kernels+2)] for m in range(num_workers*num_kernels+2)] for n in range(1000//50)] # +2 for source and sink
                    order_list = [i for i in range(num_workers)]
                    group_list = [(order_list[i], order_list[i+1]) for i in range(len(order_list)-1)]


                    ''' permutate kernel dimension with previous order '''
                    if layer_idx != 0:
                        for worker_idx in range(num_workers):
                            fix_kernel_list = []
                            for kernel_idx in range(num_kernels):
                                orig_kernel = batch_weights[worker_idx][layer_idx][kernel_idx]
                                orig_kernel = orig_kernel.reshape(orig_kernel.shape[0]//9, 3, 3)
                                fix_kernel = np.concatenate([orig_kernel[k:k+1] for k in all_kernel_order[worker_idx]])
                                fix_kernel = fix_kernel.reshape(1, orig_kernel.shape[0]*9)
                                fix_kernel_list.append(fix_kernel)
                            batch_weights[worker_idx][layer_idx] = np.concatenate(fix_kernel_list)


                    ''' assign capacity '''
                    connections = 0
                    for (worker1, worker2) in group_list:
                        for kernel1_idx in range(num_kernels):
                            for kernel2_idx in range(num_kernels):
                                s = similarity(batch_weights[worker1][layer_idx][kernel1_idx], batch_weights[worker2][layer_idx][kernel2_idx])
                                capacity_matrix_group[s//50][num_kernels*worker1+kernel1_idx][num_kernels*worker2+kernel2_idx] = s
                                
                    capacity_matrix_group = capacity_matrix_group[::-1]
                    
                    # t1 = time.time()
                    first_worker = order_list[0]
                    last_worker = order_list[-1]
                    source = num_workers*num_kernels
                    sink = num_workers*num_kernels+1
                    for kernel_idx in range(num_kernels):
                        capacity_matrix_group[0][source][num_kernels*first_worker+kernel_idx]=9999 # source to first worker
                        capacity_matrix_group[0][num_kernels*last_worker+kernel_idx][sink]=9999 # last worker to sink
                    
                    # t2 = time.time()
                    all_kernel_order = [[] for i in range(num_workers)]

                    def search_path(graph, previous_max_flow, source, sink, flow_value, global_max = 0, flow_range=None):
                        ''' trace dominate flow: dfs '''
                        path_candidate = []
                        for i in range(num_kernels*num_workers+2):
                            if graph[source][i]>0 and flow_value>previous_max_flow[i] and flow_value>global_max:
                                previous_max_flow[i] = flow_value
                                current_flow_value = min(flow_value, graph[source][i])
                                if i==sink:
                                    path_candidate.append([[], current_flow_value])
                                else:
                                    res = search_path(graph, previous_max_flow, i, sink, current_flow_value, global_max, flow_range)
                                    if res[1]!=0:
                                        res[0].append(i)
                                        if len(path_candidate)==0 or res[1]>path_candidate[0][1]:
                                            path_candidate = [[res[0], res[1]]]
                                            global_max = res[1]
                                        # path_candidate.append([res[0], res[1]])
                                        previous_max_flow = res[2]
                                        if flow_range is not None:
                                            if global_max>flow_range[0]:
                                                return path_candidate[-1][0], path_candidate[-1][1], previous_max_flow
                        
                        if len(path_candidate)!=0:
                            path_candidate = sorted(path_candidate, key=lambda x: x[1])
                            return path_candidate[-1][0], path_candidate[-1][1], previous_max_flow
                        else:
                            return [], 0, None

                    for i in range(len(capacity_matrix_group)):
                        capacity_matrix_group[i] = np.array(capacity_matrix_group[i])

                    previous_capacity_matrix = None
                    for cm_id, capacity_matrix in enumerate(capacity_matrix_group):
                        # capacity_matrix = np.array(capacity_matrix)
                        if previous_capacity_matrix is not None:
                            capacity_matrix += previous_capacity_matrix
                        
                        flag=False
                        while True:
                            capacity_graph = csr_matrix(capacity_matrix)
                            flow = maximum_flow(capacity_graph, source, sink)
                            residual_graph = flow.residual.toarray()
                            value = flow.flow_value
                            logger.info("round: {}, layer: {}, flow_value: {}".format(cr, layer_idx, value))
                            if value==0 or flag:
                                break
                            # previous_max_flow = defaultdict(lambda: -1)
                            # dominate_flow = 999
                            
                            flag = True
                            while True:
                                t1 = time.time()
                                previous_max_flow = defaultdict(lambda: -1)
                                path, dominate_flow, _ = search_path(residual_graph, previous_max_flow, source, sink, 999, 0, [950-50*cm_id, 1000-50*cm_id])
                                if dominate_flow<=950-50*cm_id:#==0:
                                    break
                                flag = False
                                path = path[::-1]
                                for wid, kid in enumerate(path):
                                    all_kernel_order[wid].append(kid%num_kernels)
                                    residual_graph[kid] = 0
                                    for i in range(len(capacity_matrix_group)):
                                        capacity_matrix_group[i][kid] = 0
                                t2 = time.time()
                                # previous_max_flow = defaultdict(lambda: -1)
                                path = [kid%num_kernels for kid in path]
                                logger.info("match: {}, dominate_flow: {}, time: {:.3f}".format(path, dominate_flow, t2-t1))
                        
                        previous_capacity_matrix = capacity_matrix
                    # t3 = time.time()

                    for worker_idx in order_list:
                        for kernel_idx in range(num_kernels):
                            if kernel_idx not in all_kernel_order[worker_idx]:
                                all_kernel_order[worker_idx].append(kernel_idx)
                        assert len(all_kernel_order[worker_idx])==num_kernels
                        batch_weights[worker_idx][layer_idx] = np.concatenate([batch_weights[worker_idx][layer_idx][k:k+1] for k in all_kernel_order[worker_idx]])

                    # t4 = time.time()

                elif layer_idx<=11 and layer_idx%2==1:
                    ''' permutate bias according to previous order '''
                    for worker_idx in range(num_workers):
                        bias = batch_weights[worker_idx][layer_idx]
                        batch_weights[worker_idx][layer_idx] = np.concatenate([bias[i:i+1] for i in all_kernel_order[worker_idx]])

                elif layer_idx==12: # first fc
                    ''' permutate bias according to previous order '''
                    for worker_idx in range(num_workers):
                        fc = batch_weights[worker_idx][layer_idx]
                        fc = fc.reshape(256, 16, 512)
                        fc = np.concatenate([fc[i:i+1] for i in all_kernel_order[worker_idx]])
                        fc = fc.reshape(4096, 512)
                        batch_weights[worker_idx][layer_idx] = fc

                
                avegerated_weight = sum([b[layer_idx] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
                averaged_weights.append(avegerated_weight)
                # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(t1-t0, t2-t1, t3-t2, t4-t3))

        else: # fedavg, fedprox
            batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
            averaged_weights = []
            num_layers = len(batch_weights[0])
            
            for i in range(num_layers):
                avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
                averaged_weights.append(avegerated_weight)

        # with open("averaged_weights_round_{}.pkl".format(cr), 'wb') as fo:
        #     pickle.dump(averaged_weights, fo)

        _ = compute_full_cnn_accuracy(None,
                            averaged_weights,
                            train_dl_global,
                            test_dl_global,
                            n_classes=None,
                            device=device,
                            args=args) # first argument is useless
        batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
        del averaged_weights
        del retrained_nets

def l2_norm(a, b):
    return np.mean((a-b)**2)**0.5

def is_match(a, b, thres=0.1):
    return True if l2_norm(a,b)<thres else False

def similarity(a, b):
    return max(int(999-l2_norm(a,b)*10000), 0)