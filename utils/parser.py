import argparse

__all__ = ['set_parser']

def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch OpenMatch Training')

    parser.add_argument('--lmbd_op', default=1.0, type=float,
                        help='pseudo label threshold')

    parser.add_argument('--mb_mask', action='store_true', default=False,
                        help='')
    parser.add_argument('--mb_op', action='store_true', default=False,
                        help='')
    parser.add_argument('--p_grad', action='store_true', default=False,
                        help='')


    parser.add_argument('--no_op', action='store_true', default=False,
                        help='')

    parser.add_argument('--p_cutoff', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--q_cutoff', default=0.5, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--dist_align', action='store_true', default=False,
                        help='')

    parser.add_argument("--no_extra_ood", action="store_true",
                        help="use var th") 
    parser.add_argument('--varth_interval', default=10, type=int,
                        help='the gradient var selection interval')

    ## selection
    parser.add_argument("--sel", action="store_true",
                        help="whether select")
    parser.add_argument("--loss_topk_ratio",default=0.1,type=float,
                        help="")
    parser.add_argument("--loss_topk", action="store_true",
                        help="whether use otsu") 
    parser.add_argument("--var_topk_ratio",default=0.1,type=float,
                        help="")
    parser.add_argument("--var_topk", action="store_true",
                        help="use topk var th")
    parser.add_argument("--loss_th", action="store_true",
                        help="whether use loss to select")
    parser.add_argument("--var_th", action="store_true",
                        help="whether use gradient var to select")
    
    ## Computational Configurations
    parser.add_argument("--ood_test", action="store_true",
                        help="whether test far ood")
    parser.add_argument("--only_id_unlabel", action="store_true",
                        help="the unlabeled dataset only contains id data")
    
    parser.add_argument('--gpu_id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--seed', default=1, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no_progress', action='store_true',
                    help="don't use progress bar")
    parser.add_argument('--eval_only', type=int, default=0,
                        help='1 if evaluation mode ')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='for cifar10')

    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--root', default='./data', type=str,
                        help='path to data directory')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'tiny'],
                        help='dataset name')
    ## Hyper-parameters
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='optimize name')
    parser.add_argument('--num_labeled', type=int, default=400,
                        choices=[50, 100, 400],
                        help='number of labeled data per each class')
    parser.add_argument('--num_val', type=int, default=50,
                        help='number of validation data per each class')
    parser.add_argument("--expand_labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet_imagenet'],
                        help='dataset name')
    ## HP unique to OpenMatch (Some are changed from FixMatch)
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                    help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                    help='coefficient of SOCR loss, 0.5 for CIFAR10, ImageNet, '
                         '1.0 for CIFAR100')
    
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--total_steps', default=262144, type=int,
                        help='number of total steps to run')
    parser.add_argument('--epochs', default=256, type=int,
                        help='number of epochs to run')
    
    ##
    parser.add_argument('--eval_step', default=1024, type=int,
                        help='number of eval steps to run')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')


    args = parser.parse_args()
    return args