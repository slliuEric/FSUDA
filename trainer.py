import time
import time

import monai.metrics
import numpy as np
import random

import scipy.stats
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import copy
import log
import sys,os
# from PyQt5.QtCore import QLibraryInfo
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
from data.dataloader import datareader,PSUdatareader,SSLdatareader
from sklearn.metrics import jaccard_score
import logging
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils_t import *
from ssim import SSIM
from NSCT import NSCT_filter
from skimage.exposure import match_histograms
import matplotlib
matplotlib.use('WebAgg')
def StyleRandomization(x, y, eps=1e-5):
    N, H, W = x.size()
    x = x.view(N, -1)
    mean_x = x.mean(-1, keepdim=True)
    var_x = x.var(-1, keepdim=True)

    # alpha = torch.rand(N, 1)
    alpha =0.1
    if x.is_cuda:
        alpha = alpha.cuda()

    y = y.view(N, -1)
    # idx_swap = torch.randperm(N)
    # y = y[idx_swap]

    # 1.use batch info
    mean_y = (y.mean(-1)).mean(-1)
    var_y = (y.var(-1)).mean(-1)

    # 2.single trgt sample info
    # mean_y = y.mean(-1, keepdim=True)
    # var_y = y.var(-1, keepdim=True)

    mean_fuse = alpha * mean_x + (1 - alpha) * mean_y
    var_fuse = alpha * var_x + (1 - alpha) * var_y

    x = (x - mean_x) / (var_x + eps).sqrt()
    x = x * (var_fuse + eps).sqrt() + mean_fuse
    x = x.view(N, H, W)

    return x, y.view(N, H, W)
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

# ==================
# init
# ==================
def trainer(args, model, snapshot_path):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('./checkpoints/' + args.exp_name):
        os.makedirs('./checkpoints/' + args.exp_name)
    if not os.path.exists('./checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('./checkpoints/' + args.exp_name + '/' + 'models')

    io = log.IOStream(args)
    # io.cprint(str(args))
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes

    # ==================
    # Read Data
    # ==================
    if args.train or args.SSLtrain:
        if args.nsct:
            src_trainset = datareader(args, args.dataroot, dataset=args.src_dataset + '_filter_trgt_hm', partition='train',
                                  domain='source')
        else:
            src_trainset = datareader(args, args.dataroot, dataset=args.src_dataset + '_train', partition='train',
                                  domain='source')
        src_valset = datareader(args, args.dataroot, dataset=args.src_dataset + '_val', partition='val',
                                domain='source')
        trgt_trainset = datareader(args, args.dataroot, dataset=args.trgt_dataset + '_train', partition='train',
                                   domain='target')
        trgt_valset = datareader(args, args.dataroot, dataset=args.trgt_dataset + '_val', partition='val',
                                 domain='target')


        src_train_loader = DataLoader(src_trainset, num_workers=0, batch_size=args.batch_size, shuffle=True,
                                      drop_last=True)
        src_val_loader = DataLoader(src_valset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
        trgt_train_loader = DataLoader(trgt_trainset, num_workers=0, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True)
        trgt_val_loader = DataLoader(trgt_valset, num_workers=0, batch_size=args.batch_size, shuffle=True,
                                     drop_last=True)
        # trgt_val_label_loader = DataLoader(trgt_val_labelset, num_workers=0, batch_size=args.batch_size, shuffle=True,
        #                                    drop_last=True)


    # ==================
    # Init Model
    # ==================
    writer = SummaryWriter(comment=args.exp_name)


    model = model.to(device)

    # Handle multi-gpu
    if (device.type == 'cuda') and len(args.gpus) > 1:
        model = nn.DataParallel(model, args.gpus)
    best_model = copy.deepcopy(model)

    # ==================
    # Optimizer
    # ==================
    opt_model = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.wd) \
        if args.optimizer == "SGD" \
        else optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.wd)
    t_max = args.epochs
    scheduler_model = CosineAnnealingLR(opt_model, T_max=t_max, eta_min=0.0)

    # ==================
    # Loss and Metrics
    # ==================

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    ssim_loss = SSIM()



    # ==================
    # Validation/test
    # ==================

    def validation(test_loader):

        # Run on cpu or gpu
        seg_loss = mIOU = accuracy = dice = asd  = 0.0
        batch_idx = num_samples = 0
        back = AA = LAC = LVC = MYO = 0.0
        back_asd = AA_asd = LAC_asd = LVC_asd = MYO_asd = 0.0
        class_dice = []
        class_asd = []
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(test_loader):
                data, labels,labels_orig = data[0].to(device), data[1].to(device), data[2].to(device)
                data = data.unsqueeze(1).type(torch.float32)
                batch_size = data.shape[0]

                outputs,_= model(data)
                loss_ce = ce_loss(outputs, labels[:].long())
                loss_dice, classwise_dice = dice_loss(outputs, labels, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                seg_loss += loss.item() * batch_size
                dice += (1-loss_dice.item()) * batch_size
                back += classwise_dice[0]
                AA  += classwise_dice[2]
                LAC += classwise_dice[4]
                LVC += classwise_dice[3]
                MYO += classwise_dice[1]

                asd_avg, classwise_asd = asd_monai(outputs, labels_orig.permute(0,3,1,2), outputs.shape[0], outputs.shape[1])
                asd+= asd_avg * batch_size
                back_asd += classwise_asd[0]
                AA_asd  += classwise_asd[2]
                LAC_asd += classwise_asd[4]
                LVC_asd += classwise_asd[3]
                MYO_asd += classwise_asd[1]

                num_samples += batch_size
                batch_idx += 1

        seg_loss /= num_samples
        dice /= num_samples
        back /= batch_idx
        AA /= batch_idx
        LAC /= batch_idx
        LVC /= batch_idx
        MYO /= batch_idx

        asd /= num_samples
        back_asd /= batch_idx
        AA_asd /= batch_idx
        LAC_asd /= batch_idx
        LVC_asd /= batch_idx
        MYO_asd /= batch_idx


        class_dice.append(back)
        class_dice.append(AA)
        class_dice.append(LAC)
        class_dice.append(LVC)
        class_dice.append(MYO)
        class_asd.append(back_asd)
        class_asd.append(AA_asd)
        class_asd.append(LAC_asd)
        class_asd.append(LVC_asd)
        class_asd.append(MYO_asd)

        return seg_loss, dice, asd, class_dice,class_asd

    def test(test_loader,model):

        # Run on cpu or gpu
        seg_loss = mIOU = accuracy = dice = asd = 0.0
        batch_idx = num_samples = 0
        back = AA = LAC = LVC = MYO = 0.0
        back_asd = AA_asd = LAC_asd = LVC_asd = MYO_asd = 0.0
        class_dice = []
        class_asd = []
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(test_loader):
                data, labels,labels_orig,name = data[0].to(device), data[1].to(device), data[3].permute(0,3,1,2).to(device),data[2]
                data = data.unsqueeze(1).type(torch.float32)
                batch_size = data.shape[0]

                outputs,_= model(data)
                if labels_orig[:,2,:,:].max() != 0:
                    print("")
                pred = torch.softmax(outputs, dim=1)
                index = pred.argmax(dim=1)
                # map = one_hot_encoder(index)12                loss_ce = ce_loss(outputs, labels[:].long())
                loss_dice, classwise_dice = dice_loss(outputs, labels, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                seg_loss += loss.item() * batch_size

                dice += (1-loss_dice.item()) * batch_size
                back += classwise_dice[0]
                AA  += classwise_dice[2]
                LAC += classwise_dice[4]
                LVC += classwise_dice[3]
                MYO += classwise_dice[1]

                asd_avg, classwise_asd = asd_monai(outputs, labels_orig, outputs.shape[0], outputs.shape[1])
                asd += asd_avg * batch_size
                back_asd += classwise_asd[0]
                AA_asd  += classwise_asd[2]
                LAC_asd += classwise_asd[4]
                LVC_asd += classwise_asd[3]
                MYO_asd += classwise_asd[1]


                num_samples += batch_size
                batch_idx += 1

        seg_loss /= num_samples
        dice /= num_samples
        asd /= num_samples
        back /= batch_idx
        AA /= batch_idx
        LAC /= batch_idx
        LVC /= batch_idx
        MYO /= batch_idx
        back_asd /= batch_idx
        AA_asd /= batch_idx
        LAC_asd /= batch_idx
        LVC_asd /= batch_idx
        MYO_asd /= batch_idx


        class_dice.append(back)
        class_dice.append(AA)
        class_dice.append(LAC)
        class_dice.append(LVC)
        class_dice.append(MYO)
        class_asd.append(back_asd)
        class_asd.append(AA_asd)
        class_asd.append(LAC_asd)
        class_asd.append(LVC_asd)
        class_asd.append(MYO_asd)

        return seg_loss,dice,asd,class_dice,class_asd



    if args.train == True:
        # ==================
        # Train
        # ==================
        src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
        src_train_dice = trgt_train_dice = src_train_asd = trgt_train_asd = 0
        src_best_val_mIOU = trgt_best_val_mIOU = 0.0
        src_best_val_loss = trgt_best_val_loss = 90000000
        epoch =  0
        mean_img = torch.zeros(1, 1)
        for epoch in range(args.epochs):
            model.train()

            # init data structures for saving epoch stats
            src_seg_loss = src_cls_loss = trgt_seg_loss = trgt_cls_loss = src_recon_loss = trgt_recon_loss = 0.0
            batch_idx = src_count = trgt_count = trgt_active_count = 0
            step = 0
            for data1, data2 in zip(src_train_loader, trgt_train_loader):
                step += 1
                opt_model.zero_grad()

                #### source data ####
                if data1 is not None:
                    src_in_trg, trg_in_src = [], []
                    if args.reconstruction:
                        src_data, src_labels, src_masked_data, src_labels_orig = data1[0], data1[1].to(device), data1[2].unsqueeze(1).to(device), data1[3].permute(0, 3, 1, 2).to(device)
                        trgt_data, trgt_labels, trgt_masked_data, trgt_labels_orig = data2[0], data2[1].to(device), \
                                                                                     data2[2].unsqueeze(1).to(device), \
                                                                                     data2[3].permute(0, 3, 1, 2).to(device)
                    else:
                        src_data, src_labels, src_labels_orig = data1[0], data1[1].to(device), data1[2].permute(0, 3, 1,2).to(device)
                        trgt_data, trgt_labels, trgt_labels_orig = data2[0], data2[1].to(device), data2[2].permute(0,3,1,2).to(device)
                    batch_size = src_data.shape[0]
                    if args.histogram_matching == True:
                        src = (src_data + 1) * 128
                        src[src > 255] = 255
                        trgt = (trgt_data + 1) * 128
                        trgt[trgt > 255] = 255
                        for i in range(batch_size):
                            st = match_histograms(np.array(src[i]), np.array(trgt[i]))
                            src_in_trg.append(st)
                        src_data = torch.tensor(src_in_trg, dtype=torch.float32)
                    if args.style_randomization == True:
                        src_data_stylerandomize, y = StyleRandomization(src_data, trgt_data)
                        # np.save('./ori_source1.npy', src_data)
                        # np.save('./trgt1.npy', y)
                        # np.save('./aft_source1.npy', src_data_stylerandomize)
                        src_data = torch.tensor(src_data_stylerandomize, dtype=torch.float32)


                    if args.fourier == True:
                        for i in range(batch_size):
                            st = FDA_source_to_target_np(src_data[i], trgt_data[i], L=0.1)
                            src_in_trg.append(st)
                        src_data = torch.tensor(src_in_trg, dtype=torch.float32)

                    if args.wavelet == True:
                        for i in range(batch_size):
                            st = WHT_transfer_wise(src_data[i], trgt_data[i],type= args.type)
                            src_in_trg.append(st)
                        src_data = torch.tensor(src_in_trg, dtype=torch.float32)

                    if args.fourier_wavelet == True:
                        for i in range(batch_size):
                            st = FDA_source_to_target_np(src_data[i], trgt_data[i], L=0.1)
                            st = WHT_transfer(st, trgt_data[i])
                            src_in_trg.append(st)
                        src_data = torch.tensor(src_in_trg, dtype=torch.float32)

                    # ##### NSCT_decompose #####
                    # levels = [1,2,3]
                    # start = time.time()
                    # src_data = NSCT_filter(src_data,levels, mask= args.mask)
                    # print("%d s used",time.time()-start)

                    src_data = src_data.unsqueeze(1).to(device).type(torch.float32)
                    batch_size = src_data.shape[0]
                    trgt_data = trgt_data.unsqueeze(1).to(device)

                    trgt_data_orig = trgt_data.clone()

                    outputs,_ = model(src_data)

                    loss_ce = ce_loss(outputs, src_labels[:].long())
                    loss_dice, classwise_dice = dice_loss(outputs, src_labels.long(), softmax=True,weight = args.diceweight)
                    triger_ent = 0.0
                    loss_ent = 0.0
                    if args.ent:
                        if step > args.switch2entropy:
                            triger_ent = 1.0
                            outputs= model(trgt_data)

                            loss_ent = ent_loss(outputs, ita=args.ita, num_class=args.num_classes)

                        # print("switch2entropy \n")

                    loss_seg = args.ceW  * loss_ce + args.diceW * loss_dice + triger_ent * args.entW * loss_ent
                    loss_seg.backward()
                    opt_model.step()

                    # print('loss_seg:',loss_seg.item())
                    src_seg_loss += loss_seg.item() * batch_size
                    src_train_dice += (1 - loss_dice.item()) * batch_size
                    # asd,hd = ASD(outputs,src_labels_orig)
                    # src_train_asd += asd * batch_size
                    src_count += batch_size
                    trgt_count += batch_size



            scheduler_model.step()

            # print progress
            src_seg_loss /= src_count
            # src_cls_loss /= src_count
            src_train_dice /= src_count
            # src_train_asd /= src_count
            # trgt_cls_loss /= trgt_count
            # src_recon_loss /= src_count
            # trgt_recon_loss /= trgt_count
            if args.active == True:
                trgt_seg_loss /= trgt_active_count
                trgt_train_dice /= trgt_active_count
                # trgt_train_asd /= trgt_active_count
            # ===================
            # Validation
            # ===================
            src_classwise_dice = trgt_classwise_dice = src_val_dice = trgt_val_dice = src_val_hd = trgt_val_hd = 0.0
            src_val_loss, src_val_dice, src_val_asd,src_val_classwise_dice, src_val_classwise_asd = validation(src_val_loader)
            trgt_val_loss, trgt_val_dice,  trgt_val_asd, trgt_val_classwise_dice, trgt_val_classwise_asd = validation(trgt_val_loader)

            # save model according to best source model (since we don't have target labels)
            if src_val_loss < src_best_val_loss:
                src_best_val_loss = src_val_loss
                # trgt_best_val_loss = trgt_val_loss
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)
            if trgt_val_loss < trgt_best_val_loss:
                trgt_best_val_loss = trgt_val_loss
                best_target_val_epoch = epoch
                best_target_model = copy.deepcopy(model)

            io.cprint(f"Epoch: {epoch}, "
                      # f"Source train cls loss: {src_cls_loss:.5f}, "
                      # f"Target train cls loss: {trgt_cls_loss:.5f}, \n"
                      # f"Source train recon loss: {src_recon_loss:.5f}, "
                      # f"Target train recon loss: {trgt_recon_loss:.5f}, \n"
                      f"Source train seg loss: {src_seg_loss:.5f}, "
                      f"Source train dice : {src_train_dice:.5f},"
                      # f"Source train asd : {src_train_asd:.5f}, \n"
                      # f"Target train seg loss: {trgt_seg_loss:.5f}, "
                      # f"Target train dice : {trgt_train_dice:.5f}, "
                      # f"Target train asd : {trgt_train_asd:.5f}, \n"
                      )
            io.cprint(f"Epoch: {epoch}, "
                      f"Source val seg loss: {src_val_loss:.5f}, "
                      f"Source val dice : {src_val_dice:.5f}, "
                      f"Source val asd : {src_val_asd:.5f}, \n"
                      f"Source val classwise_dice(back,AA,LAC,LVC,MYO) : {src_val_classwise_dice[0]:.5f}, {src_val_classwise_dice[2]:.5f},{src_val_classwise_dice[4]:.5f},{src_val_classwise_dice[3]:.5f},{src_val_classwise_dice[1]:.5f}\n"
                      f"Source val asd(back,AA,LAC,LVC,MYO) : {src_val_classwise_asd[0]:.5f}, {src_val_classwise_asd[2]:.5f},{src_val_classwise_asd[4]:.5f},{src_val_classwise_asd[3]:.5f},{src_val_classwise_asd[1]:.5f}"
                      # f"Source val hd(back,MYO,AA,LVA,LAC) : {src_val_hd[0]:.5f}, {src_val_hd[1]:.5f},{src_val_hd[2]:.5f},{src_val_hd[3]:.5f},{src_val_hd[4]:.5f}"
                      )
            io.cprint(f"Epoch: {epoch}, "
                      f"Target val seg loss: {trgt_val_loss:.5f}, "
                      f"Target val dice : {trgt_val_dice:.5f}, "
                      f"Target val asd : {trgt_val_asd:.5f}, \n"
                      f"Target val classwise_dice(back,AA,LAC,LVC,MYO) : {trgt_val_classwise_dice[0]:.5f}, {trgt_val_classwise_dice[2]:.5f},{trgt_val_classwise_dice[4]:.5f},{trgt_val_classwise_dice[3]:.5f},{trgt_val_classwise_dice[1]:.5f} \n"
                      f"Target val asd(back,AA,LAC,LVC,MYO) : {trgt_val_classwise_asd[0]:.5f}, {trgt_val_classwise_asd[2]:.5f},{trgt_val_classwise_asd[4]:.5f},{trgt_val_classwise_asd[3]:.5f},{trgt_val_classwise_asd[1]:.5f} "
                      # f"Target val asd(back,MYO,AA,LVA,LAC) : {trgt_val_hd[0]:.5f}, {trgt_val_hd[1]:.5f},{trgt_val_hd[2]:.5f},{trgt_val_hd[3]:.5f},{trgt_val_hd[4]:.5f}"
                      )
            writer.add_scalar('Source_train_seg/loss', src_seg_loss, epoch)
            writer.add_scalar('Source_train_seg/Dice', src_train_dice, epoch)
            writer.add_scalar('Source_val_seg/loss', src_val_loss, epoch)
            writer.add_scalar('Source_val_seg/Dice', src_val_dice, epoch)
            writer.add_scalar('Source_val_seg/ASD', src_val_asd, epoch)
            writer.add_scalar('Target_val_seg/loss', trgt_val_loss, epoch)
            writer.add_scalar('Target_val_seg/Dice', trgt_val_dice, epoch)
            writer.add_scalar('Target_val_seg/ASD', trgt_val_asd, epoch)

            ##save mode in each epoch##
            # checkpoint = {
            #     "model": model.state_dict(),
            #     'optimizer_model': opt_model.state_dict(),
            #     "best_val_epoch": epoch,
            #     'lr_schedule_model': scheduler_model.state_dict()
            # }
            # torch.save(checkpoint, './checkpoints/%s/models/model%s.t7' % (args.exp_name, str(epoch)))
        io.cprint("Best model was found at epoch %d\n"
                  "source val seg loss: %.4f"
                  "target val seg loss: %.4f\n"
                  % (best_val_epoch, src_best_val_loss, trgt_best_val_loss))
        checkpoint = {
            "model": best_model.state_dict(),
            'optimizer_model': opt_model.state_dict(),
            "best_val_epoch": epoch,
            'lr_schedule_model': scheduler_model.state_dict()
        }
        torch.save(checkpoint, './checkpoints/%s/models/model_best%s.t7' % (args.exp_name, str(best_val_epoch)))
        checkpoint = {
            "model": best_target_model.state_dict(),
            'optimizer_model': opt_model.state_dict(),
            "best_target_val_epoch": epoch,
            'lr_schedule_model': scheduler_model.state_dict()
        }
        torch.save(checkpoint, './checkpoints/%s/models/model_best_trgt%s.t7' % (args.exp_name, str(best_target_val_epoch)))

    else:
        # ===================
        # Test
        # ===================
        # src_valset = SSLdatareader(args, args.dataroot, dataset=args.src_dataset + '_val', partition='val', domain='source')
        # trgt_valset = SSLdatareader(args, args.dataroot, dataset=args.trgt_dataset + '_val', partition='val',domain='target')
        # src_val_loader = DataLoader(src_valset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)

        trgt_valset = SSLdatareader(args, args.dataroot, dataset='test_'+ args.trgt_dataset, partition='val',domain='target')
        trgt_val_loader = DataLoader(trgt_valset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=True)

        checkpoint = torch.load(args.saved_path)
        model.load_state_dict(checkpoint['model'])
        opt_model.load_state_dict(checkpoint['optimizer_model'])
        scheduler_model.load_state_dict(checkpoint['lr_schedule_model'])  # 加载lr_scheduler
        model.eval()

        # src_val_loss, src_val_dice, src_val_asd, src_val_classwise_dice, src_val_classwise_asd = test(src_val_loader,model)
        trgt_val_loss, trgt_val_dice,  trgt_val_asd,trgt_val_classwise_dice, trgt_val_classwise_asd = test(trgt_val_loader,model)

        # io.cprint(
        #           f"Source val seg loss: {src_val_loss:.5f}, "
        #           f"Source val dice : {src_val_dice:.5f}, "
        #           f"Source val asd : {src_val_asd:.5f}, \n"
        #           f"Source val classwise_dice(back,AA,LAC,LVC,MYO) : {src_val_classwise_dice[0]:.5f}, {src_val_classwise_dice[2]:.5f},{src_val_classwise_dice[4]:.5f},{src_val_classwise_dice[3]:.5f},{src_val_classwise_dice[1]:.5f}\n"
        #           f"Source val asd(back,AA,LAC,LVC,MYO) : {src_val_classwise_asd[0]:.5f}, {src_val_classwise_asd[2]:.5f},{src_val_classwise_asd[4]:.5f},{src_val_classwise_asd[3]:.5f},{src_val_classwise_asd[1]:.5f}"
        #           # f"Source val hd(back,MYO,AA,LVA,LAC) : {src_val_hd[0]:.5f}, {src_val_hd[1]:.5f},{src_val_hd[2]:.5f},{src_val_hd[3]:.5f},{src_val_hd[4]:.5f}"
        #           )
        io.cprint(
                  f"Target val seg loss: {trgt_val_loss:.5f}, "
                  f"Target val dice : {trgt_val_dice:.5f}, "
                  f"Target val asd : {trgt_val_asd:.5f}, \n"
                  f"Target val classwise_dice(back,AA,LAC,LVC,MYO) : {trgt_val_classwise_dice[0]:.5f}, {trgt_val_classwise_dice[2]:.5f},{trgt_val_classwise_dice[4]:.5f},{trgt_val_classwise_dice[3]:.5f},{trgt_val_classwise_dice[1]:.5f} \n"
                  f"Target val asd(back,AA,LAC,LVC,MYO) : {trgt_val_classwise_asd[0]:.5f}, {trgt_val_classwise_asd[2]:.5f},{trgt_val_classwise_asd[4]:.5f},{trgt_val_classwise_asd[3]:.5f},{trgt_val_classwise_asd[1]:.5f} "
                  # f"Target val asd(back,MYO,AA,LVA,LAC) : {trgt_val_hd[0]:.5f}, {trgt_val_hd[1]:.5f},{trgt_val_hd[2]:.5f},{trgt_val_hd[3]:.5f},{trgt_val_hd[4]:.5f}"
                  )
    # ===================
    # May be useful
    # ===================
    # if args.rotate == True:
    #     B, R, C, H, W = src_rotate_data.shape
    #     src_rotate_data = src_rotate_data.view(B * R, C, H, W)
    #     x, features = model(src_rotate_data)
    #     cls_pred = cls_head(x)
    #     loss_cls_src = ce_loss(cls_pred, src_rotate_label.view(-1).long())
    #     loss_cls_src.backward()
    #     opt_model.step()
    #     opt_rotate.step()
    #     # print('loss_cls_src:',loss_cls_src.item())
    #     src_cls_loss += loss_cls_src.item() * batch_size
    #
    #     B, R, C, H, W = trgt_rotate_data.shape
    #     trgt_rotate_data = trgt_rotate_data.view(B * R, C, H, W)
    #     x, features = model(trgt_rotate_data)
    #     cls_pred = cls_head(x)
    #     loss_cls_trgt = ce_loss(cls_pred, trgt_rotate_label.view(-1).long())
    #     loss_cls_trgt.backward()
    #     opt_model.step()
    #     opt_rotate.step()
    #     # print('loss_cls_trgt:',loss_cls_trgt.item())
    #     trgt_cls_loss += loss_cls_trgt.item() * batch_size


    ####huitu
    # import matplotlib
    # colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', '#ff9393']
    # c = matplotlib.colors.ListedColormap(colors)
    # for i in range(data.shape[0]):
    #     plt.imshow(data[i, 0, :, :].detach().cpu(), cmap='gray')
    #     plt.show()
    #     plt.imshow(labels[i, :, :].detach().cpu(), cmap=c)
    #     plt.show()
    #     plt.imshow(index[i, :, :].detach().cpu(), cmap=c)
    #     plt.show()
    ####save
    # import matplotlib
    # colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', '#ff9393']
    # c = matplotlib.colors.ListedColormap(colors)
    # i = 4
    # m = 0
    # plt.imshow(data[m, 0, :, :].detach().cpu(), cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_image%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(labels[m, :, :].detach().cpu(), cmap=c)
    # plt.axis('off')
    # plt.savefig('ct_label%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(index[m, :, :].detach().cpu(), cmap=c)
    # plt.axis('off')
    # plt.savefig('ct_pred%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # print(name[m])

    # i = 4
    # m = 7
    # plt.imshow(data[m, 0, :, :].detach().cpu(), cmap='gray')
    # plt.axis('off')
    # plt.savefig('mr_image%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(labels[m, :, :].detach().cpu(), cmap=c)
    # plt.axis('off')
    # plt.savefig('mr_label%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', '#ff9393']

    # c = matplotlib.colors.ListedColormap(colors)
    # for m in range(6):
    #     i = m
    #     plt.imshow(index[m, :, :].detach().cpu(), cmap=c)
    #     plt.axis('off')
    #     plt.savefig('1_mr_pred%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    #     print(name[m])
    #     plt.imshow(data[m, 0, :, :].detach().cpu(), cmap='gray')
    #     plt.imshow(index[m, :, :].detach().cpu(), cmap=c, alpha=0.65)
    #     plt.axis('off')
    #     plt.savefig('1_mix_mr_pred%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)

    #     plt.imshow(data[0, 0, :, :].detach().cpu(), cmap='gray')
    #     plt.axis('off')
    #     plt.savefig('ct_imaged.jpg', bbox_inches='tight', pad_inches=0, dpi=256)
    #     plt.imshow(labels[0, :, :].detach().cpu(), cmap=c)
    #     plt.axis('off')
    #     plt.savefig('ct_labeld.jpg'  , bbox_inches='tight', pad_inches=0, dpi=256)



    # colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', '#ff9393']
    # c = matplotlib.colors.ListedColormap(colors)
    # for m in range(6):
    #     i = m
    #     plt.imshow(data[0, 0, :, :].detach().cpu(), cmap='gray')
    #     plt.axis('off')
    #     plt.savefig('ct_imaged.jpg', bbox_inches='tight', pad_inches=0, dpi=256)
    #     plt.imshow(labels[0, :, :].detach().cpu(), cmap=c)
    #     plt.axis('off')
    #     plt.savefig('ct_labeld.jpg'  , bbox_inches='tight', pad_inches=0, dpi=256)
    #     plt.imshow(data[m, 0, :, :].detach().cpu(), cmap='gray')
    #     plt.imshow(index[m, :, :].detach().cpu(), cmap=c, alpha=0.65)
    #     plt.axis('off')
    #     plt.savefig('1_mix_ct_pred%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    #     plt.imshow(index[m, :, :].detach().cpu(), cmap=c)
    #     plt.axis('off')
    #     plt.savefig('1_ct_pred%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    #     print(name[m])

#purple
    # import matplotlib
    #
    # colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', ]
    # c = matplotlib.colors.ListedColormap(colors)
    # plt.imshow(labels[0, :, :].detach().cpu(), cmap=c)
    # plt.show()
    # plt.imshow(index[0, :, :].detach().cpu(), cmap=c)
    # plt.show()
####save spectrum
    # plt.imshow(A2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_A%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(H2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_H%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(V2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_V%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)
    # plt.imshow(D2, cmap='gray')
    # plt.axis('off')
    # plt.savefig('ct_D%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)

###mri
# import matplotlib
# colors = ['black', '#8ba39e', '#b4a4ca', '#fce2c4', '#ff9393']
# c = matplotlib.colors.ListedColormap(colors)
# for i in range(6):
#     plt.imshow(index[i, :, :].detach().cpu(), cmap=c)
#     plt.axis('off')
#     plt.savefig('mr1_%d.jpg' % i, bbox_inches='tight', pad_inches=0, dpi=256)