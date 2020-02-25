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

args_datadir = "./data/cifar10"


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
    if args.comm_type=='fedavg':
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
        
        logger.info("Communication round : {}".format(cr))

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
            retrained_nets = [d[rank] for rank in d]

        else:
            retrained_nets = []
            for worker_index in random.sample(range(args.n_nets), args.clients_per_round):
                dataidxs = net_dataidx_map[worker_index]
                train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 512, dataidxs)
                
                # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
                if args.comm_type=='fedavg':
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

        else: # fedavg, fedprox
            batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
            averaged_weights = []
            num_layers = len(batch_weights[0])
            
            for i in range(num_layers):
                avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
                averaged_weights.append(avegerated_weight)


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


# def get_retrained_fedavg_net(args, worker_index, dataidxs, tmp_weight, device, d, lock):
#     train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 512, dataidxs)
#     retrained_cnn = local_retrain_fedavg((train_dl_local,test_dl_local), tmp_weight, args, device=device)
#     with lock:
#         d[worker_index]=retrained_cnn.cpu()



# def fedavg_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
#                             averaging_weights, args,
#                             train_dl_global,
#                             test_dl_global,
#                             comm_round=2,
#                             device="cpu"):

#     logging.info("=="*15)
#     logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

#     for cr in range(comm_round):
        
#         logger.info("Communication round : {}".format(cr))

#         if args.multiprocess:
#             workers = random.sample(range(args.n_nets), args.clients_per_round)
#             num_workers = args.clients_per_round

#             device_list = ['cuda:%d'%(i%args.gpu_num) for i in range(num_workers)]
#             mp.set_start_method('spawn', force=True)
#             m = Manager()
#             d = m.dict()
#             lock = Lock()

#             processes = []
#             for rank in range(num_workers):
#                 p = mp.Process(target=get_retrained_fedavg_net, args=(args, rank, net_dataidx_map[rank], batch_weights[rank], device_list[rank], d, lock))
#                 p.start()
#                 processes.append(p)
#                 if len(processes)>=16: # restrict maximum processes
#                     for p in processes:
#                         p.join()
#                     processes = []
#             for p in processes:
#                 p.join()
#             retrained_nets = [d[rank] for rank in d]

#         else:
#             retrained_nets = []
#             for worker_index in random.sample(range(args.n_nets), args.clients_per_round):
#                 dataidxs = net_dataidx_map[worker_index]
#                 train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 512, dataidxs)
                
#                 # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
#                 retrained_cnn = local_retrain_fedavg((train_dl_local,test_dl_local), batch_weights[worker_index], args, device=device)
                
#                 retrained_nets.append(retrained_cnn)

#         batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

#         total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
#         fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
#         averaged_weights = []
#         num_layers = len(batch_weights[0])
        
#         for i in range(num_layers):
#             avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
#             averaged_weights.append(avegerated_weight)

#         _ = compute_full_cnn_accuracy(None,
#                             averaged_weights,
#                             train_dl_global,
#                             test_dl_global,
#                             n_classes=None,
#                             device=device,
#                             args=args)
#         batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
#         del averaged_weights

# def fedprox_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
#                             averaging_weights, args,
#                             train_dl_global,
#                             test_dl_global,
#                             comm_round=2,
#                             device="cpu"):

#     logging.info("=="*15)
#     logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

#     for cr in range(comm_round):
#         retrained_nets = []
#         logger.info("Communication round : {}".format(cr))
#         for worker_index in random.sample(range(args.n_nets), args.clients_per_round):
#             dataidxs = net_dataidx_map[worker_index]
#             train_dl_local, test_dl_local = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512, dataidxs)
            
#             # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
#             # local_retrain_fedprox(local_datasets, weights, mu, args, device="cpu")
#             retrained_cnn = local_retrain_fedprox((train_dl_local,test_dl_local), batch_weights[worker_index], mu=0.001, args=args, device=device)
            
#             retrained_nets.append(retrained_cnn)
#         batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

#         total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
#         fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
#         averaged_weights = []
#         num_layers = len(batch_weights[0])
        
#         for i in range(num_layers):
#             avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
#             averaged_weights.append(avegerated_weight)

#         _ = compute_full_cnn_accuracy(None,
#                             averaged_weights,
#                             train_dl_global,
#                             test_dl_global,
#                             n_classes=None,
#                             device=device,
#                             args=args)
#         batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
#         del averaged_weights


# def fedma_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map, 
#                             averaging_weights, args, 
#                             train_dl_global,
#                             test_dl_global,
#                             assignments_list,
#                             comm_round=2,
#                             device="cpu"):
#     '''
#     version 0.0.2
#     In this version we achieve layerwise matching with communication in a blockwise style
#     i.e. we unfreeze a block of layers (each 3 consecutive layers)---> retrain them ---> and rematch them
#     '''
#     n_layers = int(len(batch_weights[0]) / 2)
#     num_workers = len(batch_weights)

#     matching_shapes = []
#     first_fc_index = None
#     gamma = 5.0
#     sigma = 1.0
#     sigma0 = 1.0

#     cls_freqs = traindata_cls_counts
#     n_classes = args.n_class
#     batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
#     it=5

#     for cr in range(comm_round):
#         logger.info("Entering communication round: {} ...".format(cr))
#         retrained_nets = []
#         for worker_index in random.sample(range(args.n_nets), args.clients_per_round):
#             dataidxs = net_dataidx_map[worker_index]
#             train_dl_local, test_dl_local = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512, dataidxs)

#             # for the "squeezing" mode, we pass assignment list wrt this worker to the `local_retrain` function
#             recons_local_net = reconstruct_local_net(batch_weights[worker_index], args, ori_assignments=assignments_list, worker_index=worker_index)
#             retrained_cnn = local_retrain((train_dl_local,test_dl_local), recons_local_net, args,
#                                             mode="bottom-up", freezing_index=0, ori_assignments=None, device=device)
#             retrained_nets.append(retrained_cnn)

#         # BBP_MAP step
#         hungarian_weights, assignments_list = BBP_MAP(retrained_nets, model_meta_data, layer_type, net_dataidx_map, averaging_weights, args, device=device)

#         logger.info("After retraining and rematching for comm. round: {}, we measure the accuracy ...".format(cr))
#         _ = compute_full_cnn_accuracy(None,
#                                    hungarian_weights,
#                                    train_dl_global,
#                                    test_dl_global,
#                                    n_classes=None,
#                                    device=device,
#                                    args=args) # first argument is useless
#         batch_weights = [copy.deepcopy(hungarian_weights) for _ in range(args.n_nets)]
#         del hungarian_weights
#         del retrained_nets

