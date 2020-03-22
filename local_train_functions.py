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
from vgg import matched_vgg11

args_datadir = "./data/cifar10"


def trans_next_conv_layer_forward(layer_weight, next_layer_shape):
    reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
    return reshaped

def trans_next_conv_layer_backward(layer_weight, next_layer_shape):
    reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
    reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
    return reshaped

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args.dataset == "cinic10":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    losses, running_losses = [], []

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if args.dataset == "cinic10":
            scheduler.step()

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def get_local_train_net(args, worker_index, dataidxs, net, device, args_datadir, d, lock):
    logger.info("Training network %s. n_training: %d" % (str(worker_index), len(dataidxs)))
    # move the model to cuda device:
    net.to(device)

    train_dl_local, test_dl_local = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512, dataidxs)
    train_dl_global, test_dl_global = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512)
    trainacc, testacc = train_net(worker_index, net, train_dl_local, test_dl_global, args.epochs, args.lr, args, device=device)

    save_model(net, worker_index, log_dir=args.save)
    with lock:
        d[worker_index]=net.cpu()

def local_train(nets, args, net_dataidx_map, device="cpu"):
    # save local dataset
    local_datasets = []

    if not args.retrain:
        try:
            for net_id, net in nets.items():
                load_model(net, net_id, device=device, log_dir=args.pretrained_model_dir)
            nets_list = list(nets.values())
        except:
            print("fail to load models....")
            print("retraining...")
            args.retrain = True
            return local_train(nets, args, net_dataidx_map, device)

    else:
        if args.multiprocess:
            mp.set_start_method('spawn', force=True)
            m = Manager()
            d = m.dict()
            lock = Lock()
            processes = []
            for net_id, net in nets.items():
                device_idx = 'cuda:%d'%(net_id%args.gpu_num)
                p = mp.Process(target=get_local_train_net, args=(args, net_id, net_dataidx_map[net_id], net, device_idx, args_datadir, d, lock))
                # We first train the model across `num_processes` processes
                p.start()
                processes.append(p)
            
                if len(processes)>=16: # restrict maximum processes
                    for p in processes:
                        p.join()
                    processes = []
        
            for p in processes:
                p.join()
            
            nets_list = [d[rank] for rank in d]

        else:
            for net_id, net in nets.items():
                dataidxs = net_dataidx_map[net_id]
                logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
                # move the model to cuda device:
                net.to(device)

                train_dl_local, test_dl_local = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512, dataidxs)
                train_dl_global, test_dl_global = get_dataloader(args.dataset, args.args_datadir, args.batch_size, 512)

                local_datasets.append((train_dl_local, test_dl_local))

                # switch to global test set here
                trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_global, args.epochs, args.lr, args, device=device)
                # saving the trained models here
                save_model(net, net_id, log_dir=args.save)
            nets_list = list(nets.values())

    return nets_list


def local_retrain_dummy(local_datasets, weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None, device="cpu"):
    """
    FOR FPNM
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset in ("mnist", "femnist"):
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=args.n_class)
    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        if mode not in ("block-wise", "squeezing"):
            num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
            input_dim = weights[12].shape[0]
            hidden_dims = [weights[12].shape[1], weights[14].shape[1]]

            input_dim = weights[12].shape[0]
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            # so at here we need to generate a unfreezing list:
            __unfreezing_list = []
            for fi in freezing_index:
                __unfreezing_list.append(2*fi-2)
                __unfreezing_list.append(2*fi-1)

            # we need to do two changes here:
            # i) switch the number of filters in the freezing indices block to the original size
            # ii) cut the correspoidng color channels
            __fixed_indices = set([i*2 for i in range(6)]) # 0, 2, 4, 6, 8, 10
            dummy_model = ModerateCNN()

            num_filters = []
            for pi, param in enumerate(dummy_model.parameters()):
                if pi in __fixed_indices:
                    if pi in __unfreezing_list:
                        num_filters.append(param.size()[0])
                    else:
                        num_filters.append(weights[pi].shape[0])
            del dummy_model
            logger.info("################ Num filters for now are : {}".format(num_filters))
            # note that we hard coded index of the last conv layer here to make sure the dimension is compatible
            if freezing_index[0] != 6:
            #if freezing_index[0] not in (6, 7):
                input_dim = weights[12].shape[0]
            else:
                # we need to estimate the output shape here:
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                #estimated_shape = (estimated_output[1], estimated_output[2], estimated_output[3])
                input_dim = estimated_output.view(-1).size()[0]

            if (freezing_index[0] <= 6) or (freezing_index[0] > 8):
                hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
            else:
                dummy_model = ModerateCNN()
                for pi, param in enumerate(dummy_model.parameters()):
                    if pi == 2*freezing_index[0] - 2:
                        _desired_shape = param.size()[0]
                if freezing_index[0] == 7:
                    hidden_dims = [_desired_shape, weights[14].shape[1]]
                elif freezing_index[0] == 8:
                    hidden_dims = [weights[12].shape[1], _desired_shape]
        elif mode == "squeezing":
            pass


        if args.dataset in ("cifar10", "cinic10"):
            if mode == "squeezing":
                matched_cnn = ModerateCNN()
            else:
                matched_cnn = ModerateCNNContainer(3,
                                                    num_filters, 
                                                    kernel_size=3, 
                                                    input_dim=input_dim, 
                                                    hidden_dims=hidden_dims, 
                                                    output_dim=args.n_class)
        elif args.dataset in ("mnist", "femnist"):
            matched_cnn = ModerateCNNContainer(1,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=args.n_class)
    
    new_state_dict = {}
    model_counter = 0
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    #if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"):
        __non_loading_indices = []
    else:
        if mode == "block-wise":
            if freezing_index[0] != n_layers:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
                __non_loading_indices.append(__unfreezing_list[-1]+1) # add the index of the weight connects to the next layer
            else:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
        elif mode == "squeezing":
            # please note that at here we need to reconstruct the entire local network and retrain it
            __non_loading_indices = [i for i in range(len(weights))]

    def __reconstruct_weights(weight, assignment, layer_ori_shape, matched_num_filters=None, weight_type="conv_weight", slice_dim="filter"):
        # what contains in the param `assignment` is the assignment for a certain layer, a certain worker
        """
        para:: slice_dim: for reconstructing the conv layers, for each of the three consecutive layers, we need to slice the 
               filter/kernel to reconstruct the first conv layer; for the third layer in the consecutive block, we need to 
               slice the color channel 
        """
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :].T
        return res_weight

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if (param_idx in __non_loading_indices) and (freezing_index[0] != n_layers):
            # we need to reconstruct the weights here s.t.
            # i) shapes of the weights are euqal to the shapes of the weight in original model (before matching)
            # ii) each neuron comes from the corresponding global neuron
            _matched_weight = weights[param_idx]
            _matched_num_filters = weights[__non_loading_indices[0]].shape[0]
            #
            # we now use this `_slice_dim` for both conv layers and fc layers
            if __non_loading_indices.index(param_idx) != 2:
                _slice_dim = "filter" # please note that for biases, it doesn't really matter if we're going to use filter or channel
            else:
                _slice_dim = "channel"

            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="conv_weight", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_weight.reshape(param.size()))}
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)                   
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    if freezing_index[0] != 6:
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight)}
                    else:
                        # that's for handling the first fc layer that is connected to the conv blocks
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=estimated_output.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="first_fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight.T)}            
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
        else:
            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device).train()
    return matched_cnn


def local_retrain(local_datasets, weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes) ## check for  correctness
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset in ("mnist", "femnist"):
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=args.n_class)
    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        if mode not in ("block-wise", "squeezing"):
            num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
            input_dim = weights[12].shape[0]
            hidden_dims = [weights[12].shape[1], weights[14].shape[1]]

            input_dim = weights[12].shape[0]
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            # so at here we need to generate a unfreezing list:
            __unfreezing_list = []
            for fi in freezing_index:
                __unfreezing_list.append(2*fi-2)
                __unfreezing_list.append(2*fi-1)

            # we need to do two changes here:
            # i) switch the number of filters in the freezing indices block to the original size
            # ii) cut the correspoidng color channels
            __fixed_indices = set([i*2 for i in range(6)]) # 0, 2, 4, 6, 8, 10
            dummy_model = ModerateCNN()

            num_filters = []
            for pi, param in enumerate(dummy_model.parameters()):
                if pi in __fixed_indices:
                    if pi in __unfreezing_list:
                        num_filters.append(param.size()[0])
                    else:
                        num_filters.append(weights[pi].shape[0])
            del dummy_model
            logger.info("################ Num filters for now are : {}".format(num_filters))
            # note that we hard coded index of the last conv layer here to make sure the dimension is compatible
            if freezing_index[0] != 6:
            #if freezing_index[0] not in (6, 7):
                input_dim = weights[12].shape[0]
            else:
                # we need to estimate the output shape here:
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                #estimated_shape = (estimated_output[1], estimated_output[2], estimated_output[3])
                input_dim = estimated_output.view(-1).size()[0]

            if (freezing_index[0] <= 6) or (freezing_index[0] > 8):
                hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
            else:
                dummy_model = ModerateCNN()
                for pi, param in enumerate(dummy_model.parameters()):
                    if pi == 2*freezing_index[0] - 2:
                        _desired_shape = param.size()[0]
                if freezing_index[0] == 7:
                    hidden_dims = [_desired_shape, weights[14].shape[1]]
                elif freezing_index[0] == 8:
                    hidden_dims = [weights[12].shape[1], _desired_shape]
        elif mode == "squeezing":
            pass


        if args.dataset in ("cifar10", "cinic10"):
            if mode == "squeezing":
                matched_cnn = ModerateCNN()
            else:
                matched_cnn = ModerateCNNContainer(3,
                                                    num_filters, 
                                                    kernel_size=3, 
                                                    input_dim=input_dim, 
                                                    hidden_dims=hidden_dims, 
                                                    output_dim=args.n_class)
        elif args.dataset in ("mnist", "femnist"):
            matched_cnn = ModerateCNNContainer(1,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=args.n_class)
    
    new_state_dict = {}
    model_counter = 0
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    #if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"):
        __non_loading_indices = []
    else:
        if mode == "block-wise":
            if freezing_index[0] != n_layers:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
                __non_loading_indices.append(__unfreezing_list[-1]+1) # add the index of the weight connects to the next layer
            else:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
        elif mode == "squeezing":
            # please note that at here we need to reconstruct the entire local network and retrain it
            __non_loading_indices = [i for i in range(len(weights))]

    def __reconstruct_weights(weight, assignment, layer_ori_shape, matched_num_filters=None, weight_type="conv_weight", slice_dim="filter"):
        # what contains in the param `assignment` is the assignment for a certain layer, a certain worker
        """
        para:: slice_dim: for reconstructing the conv layers, for each of the three consecutive layers, we need to slice the 
               filter/kernel to reconstruct the first conv layer; for the third layer in the consecutive block, we need to 
               slice the color channel 
        """
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :].T
        return res_weight

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if (param_idx in __non_loading_indices) and (freezing_index[0] != n_layers):
            # we need to reconstruct the weights here s.t.
            # i) shapes of the weights are euqal to the shapes of the weight in original model (before matching)
            # ii) each neuron comes from the corresponding global neuron
            _matched_weight = weights[param_idx]
            _matched_num_filters = weights[__non_loading_indices[0]].shape[0]
            #
            # we now use this `_slice_dim` for both conv layers and fc layers
            if __non_loading_indices.index(param_idx) != 2:
                _slice_dim = "filter" # please note that for biases, it doesn't really matter if we're going to use filter or channel
            else:
                _slice_dim = "channel"

            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="conv_weight", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_weight.reshape(param.size()))}
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)                   
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    if freezing_index[0] != 6:
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight)}
                    else:
                        # that's for handling the first fc layer that is connected to the conv blocks
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=estimated_output.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="first_fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight.T)}            
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
        else:
            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    for param_idx, param in enumerate(matched_cnn.parameters()):
        if mode == "bottom-up":
            # for this freezing mode, we freeze the layer before freezing index
            if param_idx < freezing_index:
                param.requires_grad = False
        elif mode == "per-layer":
            # for this freezing mode, we only unfreeze the freezing index
            if param_idx not in (2*freezing_index-2, 2*freezing_index-1):
                param.requires_grad = False
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            if param_idx not in __non_loading_indices:
                param.requires_grad = False
        elif mode == "squeezing":
            pass

    matched_cnn = common_retrain(args, matched_cnn, local_datasets, device, mode=mode, freezing_index=freezing_index, weights=weights)
    return matched_cnn
    # matched_cnn.to(device).train()
    # # start training last fc layers:
    # train_dl_local = local_datasets[0]
    # test_dl_local = local_datasets[1]

    # if mode != "block-wise":
    #     if freezing_index < (len(weights) - 2):
    #         optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9)
    #     else:
    #         optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=(args.retrain_lr/10), momentum=0.9, weight_decay=0.0001)
    # else:
    #     #optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9)
    #     optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001, weight_decay=0.0001, amsgrad=True)
    
    # criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    # logger.info('n_training: %d' % len(train_dl_local))
    # logger.info('n_test: %d' % len(test_dl_local))

    # train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    # test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    # logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    # if mode != "block-wise":
    #     if freezing_index < (len(weights) - 2):
    #         retrain_epochs = args.retrain_epochs
    #     else:
    #         retrain_epochs = int(args.retrain_epochs*3)
    # else:
    #     retrain_epochs = args.retrain_epochs

    # for epoch in range(retrain_epochs):
    #     epoch_loss_collector = []
    #     for batch_idx, (x, target) in enumerate(train_dl_local):
    #         x, target = x.to(device), target.to(device)

    #         optimizer_fine_tune.zero_grad()
    #         x.requires_grad = True
    #         target.requires_grad = False
    #         target = target.long()

    #         out = matched_cnn(x)
    #         loss = criterion_fine_tune(out, target)
    #         epoch_loss_collector.append(loss.item())

    #         loss.backward()
    #         optimizer_fine_tune.step()

    #     #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
    #     epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
    #     logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    # test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    # logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    # return matched_cnn


def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3

        # YI-LIN
        elif args.dataset in ("mnist", "femnist"):
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=args.n_class)
    elif args.model == "moderate-cnn":
        matched_cnn = ModerateCNN()

    new_state_dict = {}
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn = common_retrain(args, matched_cnn, local_datasets, device)
    return matched_cnn
    # matched_cnn.to(device).train()
    # # start training last fc layers:
    # train_dl_local = local_datasets[0]
    # test_dl_local = local_datasets[1]

    # optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001, weight_decay=0.0001, amsgrad=True)
    
    # criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    # logger.info('n_training: %d' % len(train_dl_local))
    # logger.info('n_test: %d' % len(test_dl_local))

    # train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    # test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    # logger.info('>> Pre-Training Test accuracy: %f' % test_acc)


    # for epoch in range(args.retrain_epochs):
    #     epoch_loss_collector = []
    #     for batch_idx, (x, target) in enumerate(train_dl_local):
    #         x, target = x.to(device), target.to(device)

    #         optimizer_fine_tune.zero_grad()
    #         x.requires_grad = True
    #         target.requires_grad = False
    #         target = target.long()

    #         out = matched_cnn(x)
    #         loss = criterion_fine_tune(out, target)
    #         epoch_loss_collector.append(loss.item())

    #         loss.backward()
    #         optimizer_fine_tune.step()

    #     #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
    #     epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
    #     logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    # test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    # logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    # return matched_cnn


def local_retrain_fedprox(local_datasets, weights, mu, args, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    Implementing FedProx Algorithm from: https://arxiv.org/pdf/1812.06127.pdf
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset in ("mnist", "femnist"):
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=args.n_class)
    elif args.model == "moderate-cnn":
        matched_cnn = ModerateCNN()

    new_state_dict = {}
    # handle the conv layers part which is not changing
    global_weight_collector = []
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                global_weight_collector.append(torch.from_numpy(weights[param_idx].reshape(param.size())).to(device))
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                global_weight_collector.append(torch.from_numpy(weights[param_idx]).to(device))
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                global_weight_collector.append(torch.from_numpy(weights[param_idx].T).to(device))
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
                global_weight_collector.append(torch.from_numpy(weights[param_idx]).to(device))
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn = common_retrain(args, matched_cnn, local_datasets, device, global_weight_collector=global_weight_collector)
    return matched_cnn

    # matched_cnn.to(device).train()
    # # start training last fc layers:
    # train_dl_local = local_datasets[0]
    # test_dl_local = local_datasets[1]

    # optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001, weight_decay=0.0001, amsgrad=True)
    
    # criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    # logger.info('n_training: {}'.format(len(train_dl_local)))
    # logger.info('n_test: {}'.format(len(test_dl_local)))

    # train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    # test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # for epoch in range(args.retrain_epochs):
    #     epoch_loss_collector = []
    #     for batch_idx, (x, target) in enumerate(train_dl_local):
    #         x, target = x.to(device), target.to(device)

    #         optimizer_fine_tune.zero_grad()
    #         x.requires_grad = True
    #         target.requires_grad = False
    #         target = target.long()

    #         out = matched_cnn(x)
    #         loss = criterion_fine_tune(out, target)
            
    #         #########################we implement FedProx Here###########################
    #         fed_prox_reg = 0.0
    #         for param_index, param in enumerate(matched_cnn.parameters()):
    #             fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
    #         loss += fed_prox_reg
    #         ##############################################################################

    #         epoch_loss_collector.append(loss.item())

    #         loss.backward()
    #         optimizer_fine_tune.step()

    #     #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
    #     epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
    #     logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    # test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    # logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    # return matched_cnn



def reconstruct_local_net(weights, args, ori_assignments=None, worker_index=0):
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset in ("mnist", "femnist"):
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=args.n_class)
    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        matched_cnn = ModerateCNN()

        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
        # we need to estimate the output shape here:
        shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
        dummy_input = torch.rand(1, 3, 32, 32)
        estimated_output = shape_estimator(dummy_input)
        input_dim = estimated_output.view(-1).size()[0]


    def __reconstruct_weights(weight, assignment, layer_ori_shape, matched_num_filters=None, weight_type="conv_weight", slice_dim="filter"):
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :]
        return res_weight

    reconstructed_weights = []
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        _matched_weight = weights[param_idx]
        if param_idx < 1: # we need to handle the 1st conv layer specificly since the color channels are aligned
            _assignment = ori_assignments[int(param_idx / 2)][worker_index]
            _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=_assignment, 
                                                layer_ori_shape=param.size(), matched_num_filters=None,
                                                weight_type="conv_weight", slice_dim="filter")
            reconstructed_weights.append(_res_weight)

        elif (param_idx >= 1) and (param_idx < len(weights) -2):
            if "bias" in key_name: # the last bias layer is already aligned so we won't need to process it
                _assignment = ori_assignments[int(param_idx / 2)][worker_index]
                _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=_assignment, 
                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                        weight_type="bias", slice_dim=None)
                reconstructed_weights.append(_res_bias)

            elif "conv" in key_name or "features" in key_name:
                # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
                cur_assignment = ori_assignments[int(param_idx / 2)][worker_index]
                prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
                _matched_num_filters = weights[param_idx - 2].shape[0]
                _layer_ori_shape = list(param.size())
                _layer_ori_shape[0] = _matched_weight.shape[0]

                _temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                    layer_ori_shape=_layer_ori_shape, matched_num_filters=_matched_num_filters,
                                                    weight_type="conv_weight", slice_dim="channel")

                _res_weight = __reconstruct_weights(weight=_temp_res_weight, assignment=cur_assignment, 
                                                    layer_ori_shape=param.size(), matched_num_filters=None,
                                                    weight_type="conv_weight", slice_dim="filter")
                reconstructed_weights.append(_res_weight)

            elif "fc" in key_name or "classifier" in key_name:
                # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
                cur_assignment = ori_assignments[int(param_idx / 2)][worker_index]
                prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
                _matched_num_filters = weights[param_idx - 2].shape[0]

                if param_idx != 12: # this is the index of the first fc layer
                    #logger.info("%%%%%%%%%%%%%%% prev assignment length: {}, cur assignmnet length: {}".format(len(prev_assignment), len(cur_assignment)))
                    temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="fc_weight", slice_dim="channel")

                    _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                                        weight_type="fc_weight", slice_dim="filter")

                    reconstructed_weights.append(_res_weight.T)
                else:
                    # that's for handling the first fc layer that is connected to the conv blocks
                    temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                        layer_ori_shape=estimated_output.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="first_fc_weight", slice_dim=None)

                    _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                                        weight_type="fc_weight", slice_dim="filter")
                    reconstructed_weights.append(_res_weight.T)
        elif param_idx  == len(weights) - 2:
            # this is to handle the weight of the last layer
            prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
            _res_weight = _matched_weight[prev_assignment, :]
            reconstructed_weights.append(_res_weight)
        elif param_idx  == len(weights) - 1:
            reconstructed_weights.append(_matched_weight)

    return reconstructed_weights


def common_retrain(args, matched_cnn, local_datasets, device="cpu", mode=None, freezing_index=0, weights=[0], global_weight_collector=None):
    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]


    if mode is not None and mode != "block-wise":
        if freezing_index < (len(weights) - 2):
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9)
            retrain_epochs = args.retrain_epochs
        else:
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=(args.retrain_lr/10), momentum=0.9, weight_decay=0.0001)
            retrain_epochs = int(args.retrain_epochs*3)
    else:
        optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001, weight_decay=0.0001, amsgrad=True)
        retrain_epochs = args.retrain_epochs
    
    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    # logger.info('n_training: %d' % len(train_dl_local))
    # logger.info('n_test: %d' % len(test_dl_local))

    # train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    # test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    # logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    for epoch in range(args.retrain_epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, target = x.to(device), target.to(device)

            optimizer_fine_tune.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = matched_cnn(x)
            loss = criterion_fine_tune(out, target)

            ### fedprox implement
            if args.comm_type=='fedprox':
                fed_prox_reg = 0.0
                for param_index, param in enumerate(matched_cnn.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                loss += fed_prox_reg

            epoch_loss_collector.append(loss.item())

            loss.backward()
            optimizer_fine_tune.step()

        #logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        # logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    return matched_cnn