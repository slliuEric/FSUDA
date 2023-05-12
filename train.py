import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import segmentation as seg_head
# from networks.vit_seg_modeling import reconstruction as recon_head
# from networks.vit_seg_modeling import rotate_cls as cls_head
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer
from trainer_student import trainer_student
from trainer_student_contrastive import trainer_student_contrastive
from tensorboardX import SummaryWriter
# import utillog
# from utils_t import IOStream
def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--dataroot', type=str, default='/media/eric/DATA/Dataset/mmwhs', help='data path')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--rotate', type=str2bool, default=False)
parser.add_argument('--reconstruction', type=str2bool, default=False)
parser.add_argument('--fourier', type=str2bool, default=False)
parser.add_argument('--wavelet', type=str2bool, default=False)
parser.add_argument('--nsct', type=str2bool, default=False)
parser.add_argument('--fourier_wavelet', type=str2bool, default=False)
parser.add_argument('--active', type=str2bool, default=False)
parser.add_argument('--style_randomization', type=str2bool, default=False)
parser.add_argument('--histogram_matching', type=str2bool, default=True)
parser.add_argument('--MTKL', type=str2bool, default=False)
parser.add_argument('--src_dataset', type=str, default='mr', choices=['ct', 'mr'])
parser.add_argument('--trgt_dataset', type=str, default='ct', choices=['ct', 'mr'])
parser.add_argument('--rate', type=str, default='rate5', choices=['rate1', 'rate5', 'rate10', 'rate20'])
parser.add_argument('--exp_name', type=str, default='_mmwhs_hm_ct_mr_b16_e50_test',  help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--test_batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--epochs', type=int, default=50, help='number of episode to train')

parser.add_argument('--train', type=str2bool, default=False)
parser.add_argument('--SSLtrain', type=str2bool, default=False)
parser.add_argument('--ent', type=str2bool, default=False)
parser.add_argument('--saved_path', type=str, default='./checkpoints/_mmwhs_baseline_ct_mr_b16_e50/models/model14.t7', help='data path')
parser.add_argument("--entW", type=float, default=0.005, help="weight for entropy")
parser.add_argument("--ceW", type=float, default=1, help="weight for entropy")
parser.add_argument("--diceW", type=float, default=1, help="weight for segmentation loss")
parser.add_argument("--klW", type=float, default=1, help="weight for KL")
parser.add_argument("--ctW", type=float, default=1, help="weight for contrastive loss")
parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument("--diceweight", type=list, default=[1, 1, 1, 1, 1], help="weight for class_wise dice")
parser.add_argument("--ita", type=float, default=2.0, help="ita for robust entropy")
parser.add_argument("--switch2entropy", type=int, default=250, help="switch to entropy after this many steps")
parser.add_argument('--type', type=str, default='all', choices=['H', 'V','D','all'])
parser.add_argument("--mask", type=int, default=4, help="to detemine the filtered frequnency component")
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    net.load_from(weights=np.load(config_vit.pretrained_path))
    if not args.MTKL:
        trainer(args, net,snapshot_path)
    else:
        trainer_student_contrastive(args, net, snapshot_path)