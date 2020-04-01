from utils import *
import pickle
import copy
import sys
import glob
import os
import shutil
import random
import pickle
# from sklearn.preprocessing import normalize


from datasets import read_data
# from matching.gaus_marginal_matching import match_local_atoms
# from combine_nets import compute_pdm_matching_multilayer, compute_iterative_pdm_matching
from matching_performance import compute_model_averaging_accuracy, compute_full_cnn_accuracy #, compute_pdm_cnn_accuracy, compute_pdm_vgg_accuracy
from fed_functions import oneshot_matching, BBP_MAP, fed_comm
from local_train_functions import local_train



args_logdir = "logs/cifar10"
#args_dataset = "cifar10"
args_datadir = "./data/cifar10"
# args_dataroot = "/work/ntubiggg1/dataset"
args_dataroot = "/work/ntubiggg1/collabrative_files/federated_learning/leaf/data/"
# args_init_seed = 0
args_net_config = [3072, 100, 10]
#args_partition = "hetero-dir"
# args_partition = "homo"
# args_experiment = ["u-ensemble", "pdm"]
# args_trials = 1
#args_lr = 0.01
# args_epochs = 5
# args_reg = 1e-5
args_alpha = 0.5
# args_communication_rounds = 5
# args_iter_epochs=None

''' move to args '''
# args_pdm_sig = 1.0
# args_pdm_sig0 = 1.0
# args_pdm_gamma = 1.0


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lenet', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--retrain_lr', type=float, default=0.1, metavar='RLR',
                        help='learning rate using in specific for local network retrain (default: 0.01)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1, metavar='FLR',
                        help='learning rate using in specific for fine tuning the softmax layer on the data center (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained in a training process')
    parser.add_argument('--retrain_epochs', type=int, default=10, metavar='REP',
                        help='how many epochs will be trained in during the locally retraining process')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, metavar='FEP',
                        help='how many epochs will be trained in during the fine tuning process')
    parser.add_argument('--partition_step_size', type=int, default=6, metavar='PSS',
                        help='how many groups of partitions we will have')
    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')
    parser.add_argument('--partition_step', type=int, default=0, metavar='PS',
                        help='how many sub groups we are going to use for a particular training process')                          
    parser.add_argument('--n_nets', type=int, default=2, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--oneshot_matching', type=bool, default=False, metavar='OM',
                        help='if the code is going to conduct one shot matching')
    parser.add_argument('--retrain', type=bool, default=False, 
                            help='whether to retrain the model or load model locally')
    parser.add_argument('--rematching', type=bool, default=False, 
                            help='whether to recalculating the matching process (this is for speeding up the debugging process)')
    parser.add_argument('--comm_type', type=str, default='layerwise', 
                            help='which type of communication strategy is going to be used: layerwise/blockwise')    
    parser.add_argument('--comm_round', type=int, default=10, 
                            help='how many round of communications we shoud use')  
    parser.add_argument('--save', type=str, default='./tmp/checkpoints/', help='experiment path')
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument('--clients_per_round', type=int, default=-1, 
                        help='number of clients trained per round')
    parser.add_argument('--multiprocess', type=bool, default=False, help='whether to use multiprocessing')
    parser.add_argument('--gpu', type=str, default='0', help='gpu index')
    parser.add_argument('--pretrained_model_dir', type=str, default='./tmp/checkpoints/xxx/', help='load model from')
    parser.add_argument('--args_pdm_sig', type=float, default=1.0)
    parser.add_argument('--args_pdm_sig0', type=float, default=1.0)
    parser.add_argument('--args_pdm_gamma', type=float, default=1.0)
    args = parser.parse_args()
    return args

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

if __name__ == "__main__":

    args = add_fit_args(argparse.ArgumentParser(description='Probabilistic Federated CNN Matching'))
    args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpu_num = len(args.gpu.split(","))
    setattr(args, "gpu_num", gpu_num)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logger.info(device)
    
    logging.info("args = %s", args)

    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Partitioning data")

    #### YI-LIN
    # add femnist
    # changed files: main.py, utils.py, datasets.py
    if args.dataset == 'femnist':

        args_datadir = os.path.join(args_dataroot, args.dataset, 'data')

        users, _, data = read_data(os.path.join(args_datadir, 'train'))

        y_train = []

        net_dataidx_map = {}
        pre = 0
        for i, (_, _data) in enumerate(data.items()):
            y_train += _data['y']
            net_dataidx_map[i] = np.arange(pre, pre + len(_data['y']))
            pre += len(_data['y'])

            # if i > 3:
            #     break
        
        y_train = np.hstack(y_train)

        # y_train = np.hstack(y_train)

        # net_dataidx_map = {i:u for i, u in enumerate(users)}

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, None)
        setattr(args, "traindata_cls_counts", traindata_cls_counts)

        train_dl_global, test_dl_global = get_dataloader(args.dataset, args_datadir, args.batch_size, 512)

        setattr(args, "n_nets", len(net_dataidx_map))
        

    else:
        if args.partition != "hetero-fbs":
            y_train, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args_datadir, args_logdir, 
                                                                    args.partition, args.n_nets, args_alpha, args=args)
            setattr(args, "traindata_cls_counts", traindata_cls_counts)
        else:
            y_train, net_dataidx_map, traindata_cls_counts, baseline_indices = partition_data(args.dataset, args_datadir, args_logdir, 
                                                        args.partition, args.n_nets, args_alpha, args=args)
            setattr(args, "traindata_cls_counts", traindata_cls_counts)

        train_dl_global, test_dl_global = get_dataloader(args.dataset, args_datadir, args.batch_size, 512)

    setattr(args, "args_datadir", args_datadir)

    # YI-LIN
    print("num of clients: ", args.n_nets)
    if args.clients_per_round == -1:
        setattr(args, "clients_per_round", args.n_nets)

    n_classes = len(np.unique(y_train))
    setattr(args, 'n_class', n_classes)
    
    averaging_weights = np.zeros((args.n_nets, n_classes), dtype=np.float32)

    for i in range(n_classes):
        total_num_counts = 0
        worker_class_counts = [0] * args.n_nets
        for j in range(args.n_nets):
            if i in args.traindata_cls_counts[j].keys():
                total_num_counts += args.traindata_cls_counts[j][i]
                worker_class_counts[j] = args.traindata_cls_counts[j][i]
            else:
                total_num_counts += 0
                worker_class_counts[j] = 0
        averaging_weights[:, i] = worker_class_counts / total_num_counts

    logger.info("averaging_weights: {}".format(averaging_weights))

    logger.info("Initializing nets")
    nets, model_meta_data, layer_type = init_models(args_net_config, args.n_nets, args)
    logger.info("Retrain? : {}".format(args.retrain))

    ### local training stage
    nets_list = local_train(nets, args, net_dataidx_map, device=device)

    # # ensemble part of experiments
    # logger.info("Computing Uniform ensemble accuracy")
    # uens_train_acc, _ = compute_ensemble_accuracy(nets_list, train_dl_global, n_classes,  uniform_weights=True, device=device)
    # uens_test_acc, _ = compute_ensemble_accuracy(nets_list, test_dl_global, n_classes, uniform_weights=True, device=device)

    # logger.info("Uniform ensemble (Train acc): {}".format(uens_train_acc))
    # logger.info("Uniform ensemble (Test acc): {}".format(uens_test_acc))

    # # for PFNM
    # if args.oneshot_matching:
    #     hungarian_weights, assignments_list = oneshot_matching(nets_list, model_meta_data, layer_type, net_dataidx_map, averaging_weights, args, device=device)
    #     _ = compute_full_cnn_accuracy(hungarian_weights,
    #                                hungarian_weights,
    #                                train_dl_global,
    #                                test_dl_global,
    #                                n_classes,
    #                                device=device,
    #                                args=args)

    # # this is for PFNM
    # hungarian_weights, assignments_list = BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map, averaging_weights, args, device=device)

    ## averaging models 
    ## we need to switch to real FedAvg implementation 
    ## FedAvg is originally proposed at: here: https://arxiv.org/abs/1602.05629
    batch_weights = pdm_prepare_full_weights_cnn(nets_list, device=device)
    #dataidxs = net_dataidx_map[args.rank]
    total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
    logger.info("Total data points: {}".format(total_data_points))
    logger.info("Freq of FedAvg: {}".format(fed_avg_freqs))

    averaged_weights = []
    num_layers = len(batch_weights[0])
    for i in range(num_layers):
        avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
        averaged_weights.append(avegerated_weight)

    for aw in averaged_weights:
        logger.info(aw.shape)

    # with open("hungarian_weights.pkl", 'wb') as fo:
    #     pickle.dump(hungarian_weights, fo)
    # with open("averaged_weights.pkl", 'wb') as fo:
    #     pickle.dump(averaged_weights, fo)
    # with open("assignments_list.pkl", 'wb') as fo:
    #     pickle.dump(assignments_list, fo)

    # with open("hungarian_weights.pkl", "rb") as fi:
    #     hungarian_weights = pickle.load(fi)
    # with open("averaged_weights.pkl", 'rb') as fi:
    #     averaged_weights = pickle.load(fi)
    # with open("assignments_list.pkl", 'rb') as fi:
    #     assignments_list = pickle.load(fi)
    
    # print(args)

    models = nets_list
    # _ = compute_full_cnn_accuracy(models, hungarian_weights, train_dl_global, test_dl_global, n_classes, device, args)

    # _ = compute_model_averaging_accuracy(models, averaged_weights, train_dl_global, test_dl_global, n_classes, args)

    if args.comm_type=="fedma":
        comm_init_batch_weights = [copy.deepcopy(hungarian_weights) for _ in range(args.n_nets)]
    else: # fedavg, fedprox
        comm_init_batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
        assignments_list = None
    
    fed_comm(comm_init_batch_weights, model_meta_data, layer_type, net_dataidx_map, averaging_weights, args,
             train_dl_global, test_dl_global, comm_round=args.comm_round, device=device, assignments_list=assignments_list)
