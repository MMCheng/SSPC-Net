# encoding=utf-8
from __future__ import division
from __future__ import print_function
from builtins import range

import time
import random
import numpy as np
import json
import os
import sys
import math
import ast
import argparse
from collections import defaultdict
import h5py
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torchnet as tnt
from extension import extension_accum2
import torch.nn.functional as F

import spg
import graphnet
import pointnet
import metrics
import attention_ext_module
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True, linewidth=220)
import warnings
warnings.filterwarnings("ignore")



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=10, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=10, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_multisamp_n', default=10, type=int, help='Average logits obtained over runs with different seeds')

    # Dataset
    parser.add_argument('--dataset', default='s3dis', help='Dataset name: sema3d|s3dis')
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting (S3DIS)')
    parser.add_argument('--odir', default='results', help='Directory to store results')
    parser.add_argument('--resume', default='', help='Loads a previously saved model.')
    parser.add_argument('--db_train_name', default='train')
    parser.add_argument('--db_test_name', default='val')
    parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3d')
    parser.add_argument('--S3DIS_PATH', default='datasets/s3dis')
    parser.add_argument('--SCANNET_PATH', default='datasets/scannet')
    parser.add_argument('--VKITTI_PATH', default='datasets/vkitti')
    parser.add_argument('--CUSTOM_SET_PATH', default='datasets/custom_set')

    # Model
    parser.add_argument('--model_config', default='gru_10,f_8', help='Defines the model as a sequence of layers, see graphnet.py for definitions of respective layers and acceptable arguments. In short: rectype_repeats_mv_layernorm_ingate_concat, with rectype the type of recurrent unit [gru/crf/lstm], repeats the number of message passing iterations, mv (default True) the use of matrix-vector (mv) instead vector-vector (vv) edge filters, layernorm (default True) the use of layernorms in the recurrent units, ingate (default True) the use of input gating, concat (default True) the use of state concatenation')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--edge_attribs', default='delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d', help='Edge attribute definition, see spg_edge_features() in spg.py for definitions.')

    # Point cloud processing
    parser.add_argument('--pc_attribs', default='', help='Point attributes fed to PointNets, if empty then all possible.')
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=1, type=int, help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--pc_xyznormalize', default=1, type=int, help='Bool, normalize xyz into unit ball, i.e. in [-0.5,0.5]')

    # Filter generating network
    parser.add_argument('--fnet_widths', default='[32,128,64]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=0, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    parser.add_argument('--fnet_bnidx', default=2, type=int, help='Layer index to insert batchnorm to. -1=do not insert.')
    parser.add_argument('--edge_mem_limit', default=30000, type=int, help='Number of edges to process in parallel during computation, a low number can reduce memory peaks.')

    # Superpoint graph
    parser.add_argument('--spg_attribs01', default=1, type=int, help='Bool, normalize edge features to 0 mean 1 deviation')
    parser.add_argument('--spg_augm_nneigh', default=100, type=int, help='Number of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_order', default=3, type=int, help='Order of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_hardcutoff', default=512, type=int, help='Maximum number of superpoints larger than args.ptn_minpts to sample in SPG')
    parser.add_argument('--spg_superedge_cutoff', default=-1, type=float, help='Artificially constrained maximum length of superedge, -1=do not constrain')

    # Point net
    parser.add_argument('--ptn_minpts', default=40, type=int, help='Minimum number of points in a superpoint for computing its embedding.')
    parser.add_argument('--ptn_npts', default=128, type=int, help='Number of input points for PointNet.')
    parser.add_argument('--ptn_widths', default='[[64,64,128,128,256], [256,64,32]]', help='PointNet widths')
    parser.add_argument('--ptn_widths_stn', default='[[64,64,128], [128,64]]', help='PointNet\'s Transformer widths')
    parser.add_argument('--ptn_nfeat_stn', default=11, type=int, help='PointNet\'s Transformer number of input features')
    parser.add_argument('--ptn_prelast_do', default=0, type=float)
    parser.add_argument('--ptn_mem_monger', default=1, type=int, help='Bool, save GPU memory by recomputing PointNets in back propagation.')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')

    # extension
    parser.add_argument('--extension_th', type=float, default=0.95)
    parser.add_argument('--loss_w1', type=float, default=1.0, help='weight of cross entropy loss')
    parser.add_argument('--loss_w2', type=float, default=1.0, help='weight of cross entropy loss of extension')
    parser.add_argument('--ext_epoch_gap', type=int, default=40, help='interval of epoch for accumulated extension')
    parser.add_argument('--ext_drop', type=float, default=0.9, help='dropout ratio for accumulated extension')
    parser.add_argument('--single_ext_max', type=int, default=40, help='To reduce memory cost,maximum points for a single extension')
    parser.add_argument('--max_labeled_att', type=int, default=400, help='To reduce memory cost, maximum labeled points for extension attention')
    parser.add_argument('--max_ext_att_loss', type=int, default=350, help='To reduce memory cost, maximum extension points participated in attention and loss')
    
    parser.add_argument('--data_mode', type=str, default='voxel', choices=['point', 'voxel'])
    parser.add_argument('--train_gpu', type=int, default=0, help='ignore label')
    parser.add_argument('--metric_ignore_class', type=int, default=None, help='maximum extension points participated in attention and loss')


    args = parser.parse_args()
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    args.ptn_widths = ast.literal_eval(args.ptn_widths)
    args.ptn_widths_stn = ast.literal_eval(args.ptn_widths_stn)
    args.extension_dir = os.path.join(args.odir, 'extension_log') # To store the extension intermediate results

    print(args)


    print('Will save to ' + args.odir)
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    if not os.path.exists(args.extension_dir):
        os.makedirs(args.extension_dir)
    with open(os.path.join(args.odir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(["'"+a+"'" if (len(a)==0 or a[0]!='-') else a for a in sys.argv]))

    set_seed(args.seed, args.cuda)
    if (args.dataset=='sema3d' and args.db_test_name.startswith('test')) or (args.dataset.startswith('s3dis_02') and args.cvfold==2):
        # needed in pytorch 0.2 for super-large graphs with batchnorm in fnet  (https://github.com/pytorch/pytorch/pull/2919)
        torch.backends.cudnn.enabled = False


    # Decide on the dataset
    if args.dataset=='s3dis':
        import s3dis_dataset
        dbinfo = s3dis_dataset.get_info(args)
        create_dataset = s3dis_dataset.get_datasets
        args.n_labels = 13
    elif args.dataset=='scannet':
        import scannet_dataset
        dbinfo = scannet_dataset.get_info(args)
        create_dataset = scannet_dataset.get_datasets
        args.n_labels = 20
    elif args.dataset=='vkitti':
        import vkitti_dataset
        dbinfo = vkitti_dataset.get_info(args)
        create_dataset = vkitti_dataset.get_datasets
        args.n_labels = 13

    elif args.dataset=='custom_dataset':
        import custom_dataset #<- to write!
        dbinfo = custom_dataset.get_info(args)
        create_dataset = custom_dataset.get_datasets
    else:
        raise NotImplementedError('Unknown dataset ' + args.dataset)

    # Create model and optimizer
    if args.resume != '':
        model, optimizer, stats = resume(args, dbinfo)
    else:
        model = create_model(args, dbinfo)
        optimizer = create_optimizer(args, model)
        stats = []


    train_dataset, test_dataset = create_dataset(args)
    ptnCloudEmbedder = pointnet.CloudEmbedder(args)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay, last_epoch=args.start_epoch-1)


    ############
    def train(epoch):
        """ Trains for one epoch """
        args.ext_epoch = epoch
        model.train()

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=spg.eccpc_collate, num_workers=args.nworkers, 
                    shuffle=True, drop_last=True)

        loss_meter = tnt.meter.AverageValueMeter()
        loss_ext_meter = tnt.meter.AverageValueMeter()
        loss_att_meter = tnt.meter.AverageValueMeter()
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'], ignore_label = args.metric_ignore_class)
        confusion_matrix_ext = metrics.ConfusionMatrix(dbinfo['classes'], ignore_label = args.metric_ignore_class)
        confusion_matrix_ext_epoch = metrics.ConfusionMatrix(dbinfo['classes'], ignore_label = args.metric_ignore_class)
        t0 = time.time()
        epoch_time = time.time()
        batch_time = AverageMeter()
        end = time.time()

        # iterate over dataset in batches
        for bidx, (targets, GIs, clouds_data, clouds_orig, edges_for_ext, fnames, ext_data, num_sp_list) in enumerate(loader):
            print('fnames: {}'.format(fnames))
            t_loader = time.time()-t0

            model.ecc.set_info(GIs, args.cuda)
            weak_label_mode_cpu, label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,1], targets[:,2:], targets[:,2:].sum(1)
            ext_mask, extension_sub_list, extension_full_list = ext_data

            if args.cuda:
                label_mode, label_vec, segm_size, weak_label_mode = label_mode_cpu.cuda(), label_vec_cpu.float().cuda(), segm_size_cpu.float().cuda(), weak_label_mode_cpu.cuda()
            else:
                label_mode, label_vec, segm_size, weak_label_mode = label_mode_cpu, label_vec_cpu.float(), segm_size_cpu.float(), weak_label_mode_cpu

            optimizer.zero_grad()
            t0 = time.time()

            
            print('num_weak_label/num_sp_all: {}/{}'.format(torch.sum(weak_label_mode<args.n_labels+1), weak_label_mode.shape[0]))

            embeddings = ptnCloudEmbedder.run(model, *clouds_data)
            outputs, rnn_fea = model.ecc(embeddings)
            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())

            # extension
            input = clouds_orig.cuda()
            edges_for_ext = edges_for_ext.cuda()
            ext_weak_label_cuda = torch.argmax(outputs, dim=1, keepdim=False)

            weak_label_cat = weak_label_mode
            weak_label_cat[ext_mask>0] = ext_weak_label_cuda[ext_mask>0]

            if (epoch>0) and (epoch % args.ext_epoch_gap == 0):
                output1, weak_label1, output2, weak_label2, extend_idx, _ = extension_accum2(input, outputs, embeddings, weak_label_cat, edges_for_ext, th=args.extension_th, ext_max=args.single_ext_max)
                print('{}/{} ~ {:.2f} points with labels.'.format(outputs.shape[0], output1.shape[0], output1.shape[0]/outputs.shape[0]*100))
                print('extension points: {}'.format(extend_idx.shape))
            else:
                extend_idx = torch.Tensor([])
                output2 = torch.Tensor([])
                weak_label2 = torch.Tensor([])


            # loss
            mask1 = weak_label_mode != args.ignore_label
            outputs_valid1 = outputs[mask1, :]
            weak_label1  = weak_label_mode[mask1]
            loss1_cro = nn.functional.cross_entropy(outputs_valid1, weak_label1, ignore_index=args.ignore_label)


            ###### extension dropout
            # labeled points: outputs_valid1, weak_label1
            labeled_idx = torch.nonzero(mask1).squeeze().long() # Nsp_label
            # previous extension points:
            if torch.sum(ext_mask>0) > 1:
                pre_ext_outputs = outputs[ext_mask>0, :]    # Nsp_pre_ext
                pre_ext_fea = rnn_fea[ext_mask>0, :] 
                pre_ext_pseudo_label = torch.argmax(pre_ext_outputs, dim=1, keepdim=False)
                pre_ext_idx = torch.nonzero(ext_mask>0).squeeze().long()
            # current extension points:
            if extend_idx.shape[0]>1:
                cur_ext_outputs = output2
                cur_ext_pred_label = weak_label2
                cur_ext_idx = extend_idx
                cur_ext_fea = rnn_fea[extend_idx, :] 

            # extension concat
            if (torch.sum(ext_mask>0) > 1) & (extend_idx.shape[0]>1):
                ext_outputs_cat = torch.cat((pre_ext_outputs, cur_ext_outputs), 0)
                ext_label_cat = torch.cat((pre_ext_pseudo_label.unsqueeze(-1), cur_ext_pred_label.unsqueeze(-1)), 0).squeeze(-1)
                ext_idx_cat = torch.cat((pre_ext_idx.unsqueeze(-1), cur_ext_idx.unsqueeze(-1)), 0).squeeze(-1)
                ext_fea_cat = torch.cat((pre_ext_fea, cur_ext_fea), 0)
            elif extend_idx.shape[0] > 1: # only current extension points
                ext_outputs_cat = cur_ext_outputs
                ext_label_cat = cur_ext_pred_label
                ext_idx_cat = cur_ext_idx
                ext_fea_cat = cur_ext_fea
            elif torch.sum(ext_mask>0) > 1: # only previous extension points
                ext_outputs_cat = pre_ext_outputs
                ext_label_cat = pre_ext_pseudo_label
                ext_idx_cat = pre_ext_idx
                ext_fea_cat = pre_ext_fea
            else:
                ext_outputs_cat = None
                ext_label_cat = None
                ext_idx_cat = None
                ext_fea_cat = None


            # compute the cluster center and the distances and then dropout with ratio
            if (ext_idx_cat is not None) and (ext_idx_cat.shape[0]>20):
                ext_idxs_sample = [] # sampled id of each extension points in all the extension points
                unique_classes = torch.unique(weak_label1)
                ext_idx_idx = torch.Tensor(list(range(ext_idx_cat.shape[0]))).cuda() # id of each extension points in all the extension points
                for i in range(unique_classes.shape[0]):
                    sp = unique_classes[i]
                    fea_label = outputs_valid1[weak_label1==sp] # Nsp_label_class*13
                    fea_ext = ext_outputs_cat[ext_label_cat==sp] # Nsp_ext_class*13
                    ext_idxs = ext_idx_idx[ext_label_cat==sp]  # Nsp_ext_class

                    if (fea_ext.shape[0] > 5) & (fea_label.shape[0] > 0):
                        num_retain = math.floor(fea_ext.shape[0] * args.ext_drop)

                        cluster_center = torch.sum(fea_label, dim=0, keepdim=True) + 0.5 * torch.sum(fea_ext, dim=0, keepdim=True)
                        cluster_center = cluster_center / (fea_label.shape[0] + fea_ext.shape[0]) # 1*13

                        dis = fea_ext - cluster_center # Nsp_ext*13
                        dis = torch.norm(dis, dim=1) # Nsp
                        _, idxs = torch.sort(dis, dim=0, descending=False)
                        ext_idxs_sample.append(ext_idxs[idxs[:num_retain]].unsqueeze(-1))
                    elif len(ext_idxs) > 0:
                        ext_idxs_sample.append(ext_idxs.unsqueeze(-1))

                ext_idxs_sample = torch.cat(ext_idxs_sample, 0).squeeze(-1).long()

                ext_idxs_retain = ext_idx_cat[ext_idxs_sample]
                ext_output_retain = ext_outputs_cat[ext_idxs_sample, :]
                ext_fea_retain = ext_fea_cat[ext_idxs_sample, :]
                ext_label_retain = ext_label_cat[ext_idxs_sample]

            else:
                ext_idxs_retain = ext_idx_cat
                ext_output_retain = ext_outputs_cat
                ext_fea_retain = ext_fea_cat
                ext_label_retain = ext_label_cat


            if (ext_idxs_retain is not None) and (ext_idxs_retain.shape[0] > 2):
                lab_fea = rnn_fea[labeled_idx, :]   # M*352, including the previous extension points
                lab_lab = weak_label1
                if lab_fea.shape[0] > args.max_labeled_att:
                    ii = random.sample(range(lab_fea.shape[0]), k=args.max_labeled_att)
                    lab_fea = lab_fea[ii, :]
                    lab_lab = lab_lab[ii]
                    lab_idxs_sample = ii
                if ext_idxs_retain.shape[0] > args.max_ext_att_loss:
                    ii = random.sample(range(ext_idxs_retain.shape[0]), k=args.max_ext_att_loss)
                    # ext_idxs_retain_sample = ext_idxs_retain[ii]
                    ext_idxs_retain_sample = ii
                else:
                    # ext_idxs_retain_sample = ext_idxs_retain
                    ext_idxs_retain_sample = list(range(ext_idxs_retain.shape[0]))
                ext_fea = rnn_fea[ext_idxs_retain_sample, :]   # N*352

                # coupled attention
                outputs_att_lab = model.att_lab(lab_fea, ext_fea)
                outputs_att_ext = model.att_ext(ext_fea, lab_fea)

                loss_att_lab_cro = nn.functional.cross_entropy(outputs_att_lab, lab_lab)
                loss_att_meter.add(loss_att_lab_cro.item())

                loss_att_ext_cro = nn.functional.cross_entropy(outputs_att_ext, ext_label_retain[ext_idxs_retain_sample])
                loss_ext_meter.add(loss_att_ext_cro.item())
                
                loss = args.loss_w1 * loss1_cro + args.loss_w2 * (loss_att_lab_cro + loss_att_ext_cro)

                confusion_matrix_ext.count_predicted_batch(tvec_cpu[ext_idxs_retain.data.cpu().numpy(), :], np.argmax(outputs[ext_idxs_retain, :].data.cpu().numpy(),1))
                print('{} point extend. acc: {:3f}, macc: {:3f}'.format(ext_idxs_retain.shape[0], confusion_matrix_ext.get_overall_accuracy(), confusion_matrix_ext.get_mean_class_accuracy()))
                if extend_idx.shape[0]>1:
                    confusion_matrix_ext_epoch.count_predicted_batch(tvec_cpu[extend_idx.data.cpu().numpy(), :], np.argmax(outputs[extend_idx, :].data.cpu().numpy(),1))
                    print('{} point extend current epoch. acc: {:3f}, macc: {:3f}'.format(extend_idx.shape[0], confusion_matrix_ext_epoch.get_overall_accuracy(), confusion_matrix_ext_epoch.get_mean_class_accuracy()))


                # update the extension
                extend_idx = ext_idxs_retain.data.cpu().numpy().astype(np.int32)
                weak_label2 = ext_label_retain.data.cpu().numpy().astype(np.int32)
                num_sp_array = np.cumsum(np.array(num_sp_list))

                for b in range(num_sp_array.shape[0]):
                    if b==0:
                        sp_start = 0
                    else:
                        sp_start = num_sp_array[b-1]
                    sp_end = num_sp_array[b]

                    mask = (extend_idx>=sp_start) & (extend_idx<sp_end)
                    if np.sum(mask) > 0:
                        extend_idx_batch = extend_idx[mask] - sp_start
                        extend_label_batch = weak_label2[mask]

                        extension_sub_batch = extension_sub_list[b].astype(np.int32)     # Nsp*2
                        extension_full_batch = extension_full_list[b].astype(np.int32)   # N*2


                        extension_full_batch[extension_sub_batch[extend_idx_batch, 0]] = extend_label_batch
                        current_fname = fnames[b]
                        np.savetxt(os.path.join(args.extension_dir, 'epoch_{:d}'.format(int(epoch//args.ext_epoch_gap)), '{}.txt'.format(current_fname)), extension_full_batch, fmt='%d')

            else:
                loss = loss1_cro


            loss.backward()
            ptnCloudEmbedder.bw_hook()

            if args.grad_clip>0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            optimizer.step()

            t_trainer = time.time()-t0
            # loss_meter.add(loss.data[0]) # pytorch 0.3
            loss_meter.add(loss.item()) # pytorch 0.4


            acc_meter.add(o_cpu, t_cpu)
            confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))

            batch_time.update(time.time() - end)
            end = time.time()

            print('Batch {}/{} - loss {:.3f}/{:.3f}, acc {:.3f}, lr {:.3f}, Loader time {:.3f}, Trainer time {:.3f}, Batch time {:.3f}/{:.3f}.'.format(bidx+1, len(loader), 
                loss.item(), loss_meter.value()[0], confusion_matrix.get_overall_accuracy(), get_lr(optimizer), t_loader, t_trainer, batch_time.val, batch_time.avg))
            t0 = time.time()

        if args.ext_epoch%args.ext_epoch_gap == (args.ext_epoch_gap-1): # a new extension folder need to be built
            shutil.copytree(os.path.join(args.extension_dir, 'epoch_{}'.format(int(args.ext_epoch//args.ext_epoch_gap))), 
                            os.path.join(args.extension_dir, 'epoch_{}'.format(int(args.ext_epoch//args.ext_epoch_gap)+1)))
        return acc_meter.value()[0], confusion_matrix.get_overall_accuracy(), confusion_matrix.get_mean_class_accuracy(), confusion_matrix.get_average_intersection_union(), loss_meter.value()[0], time.time()-epoch_time, \
                    confusion_matrix_ext_epoch.get_overall_accuracy(), confusion_matrix_ext_epoch.get_mean_class_accuracy(), confusion_matrix_ext_epoch.get_average_intersection_union()

    ############
    def eval():
        """ Evaluated model on test set """
        model.eval()

        loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=spg.eccpc_collate_test, num_workers=args.nworkers, drop_last=False)

        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'], ignore_label = args.metric_ignore_class)
        test_time = time.time()
        # iterate over dataset in batches
        for bidx, (targets, GIs, clouds_data, clouds_orig, edges_for_ext, fnames) in enumerate(loader):
            model.ecc.set_info(GIs, args.cuda)
            # label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
            weak_label_mode_cpu, label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,1], targets[:,2:], targets[:,2:].sum(1)

            embeddings = ptnCloudEmbedder.run(model, *clouds_data)
            outputs, _ = model.ecc(embeddings)


            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())
            if t_cpu.size > 0:
                acc_meter.add(o_cpu, t_cpu)
                confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))
                print('{}/{}-{} outputs: {}, acc: {}, macc: {}'.format(bidx, len(loader), fnames[0], outputs.shape, confusion_matrix.get_overall_accuracy(), confusion_matrix.get_mean_class_accuracy()))


        return meter_value(acc_meter), confusion_matrix.get_overall_accuracy(), confusion_matrix.get_mean_class_accuracy(), confusion_matrix.get_average_intersection_union(), time.time()-test_time

    ############
    def eval_final():
        """ Evaluated model on test set in an extended way: computes estimates over multiple samples of point clouds and stores predictions """
        model.eval()

        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(dbinfo['classes'], ignore_label = args.metric_ignore_class)
        collected, predictions = defaultdict(list), {}

        # collect predictions over multiple sampling seeds
        for ss in range(args.test_multisamp_n):
            test_dataset_ss = create_dataset(args, ss)[1]
            loader = torch.utils.data.DataLoader(test_dataset_ss, batch_size=1, collate_fn=spg.eccpc_collate_test, num_workers=args.nworkers)

            # iterate over dataset in batches
            for bidx, (targets, GIs, clouds_data, clouds_orig, edges_for_ext, _) in enumerate(loader):
                model.ecc.set_info(GIs, args.cuda)
                weak_label_mode_cpu, label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,1], targets[:,2:], targets[:,2:].sum(1)

                embeddings = ptnCloudEmbedder.run(model, *clouds_data)
                outputs, _ = model.ecc(embeddings)

                fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
                collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))

        # aggregate predictions (mean)
        for fname, lst in collected.items():
            o_cpu, t_cpu, tvec_cpu = list(zip(*lst))
            if args.test_multisamp_n > 1:
                o_cpu = np.mean(np.stack(o_cpu,0),0)
            else:
                o_cpu = o_cpu[0]
            t_cpu, tvec_cpu = t_cpu[0], tvec_cpu[0]
            predictions[fname] = np.argmax(o_cpu,1)
            o_cpu, t_cpu, tvec_cpu = filter_valid(o_cpu, t_cpu, tvec_cpu)
            if t_cpu.size > 0:
                acc_meter.add(o_cpu, t_cpu)
                confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))

        per_class_iou = {}
        perclsiou = confusion_matrix.get_intersection_union_per_class()
        for c, name in dbinfo['inv_class_map'].items():
            per_class_iou[name] = perclsiou[c]

        return meter_value(acc_meter), confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union(), per_class_iou, predictions,  confusion_matrix.get_mean_class_accuracy(), confusion_matrix.confusion_matrix

    ############
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.odir))
        # scheduler.step()

        _, acc, macc, miou, loss, epoch_time, ext_acc, ext_macc, ext_miou = train(epoch)

        if (epoch+1) % args.test_nth_epoch == 0 or epoch+1==args.epochs:
            _, test_acc, test_macc, test_miou, test_time = eval()
            print('-> [{}/{}] Train results as epoch: mIou/mAcc/allAcc/loss/time {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f} '.format(epoch+1, args.epochs, miou, macc, acc, loss, epoch_time))
            print('-> [{}/{}] Train extension results as epoch: mIou/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}/ '.format(epoch+1, args.epochs, ext_miou, ext_macc, ext_acc))
            print('-> [{}/{}] Test results as epoch: mIou/mAcc/allAcc/time {:.4f}/{:.4f}/{:.4f}/{:.4f} '.format(epoch+1, args.epochs, test_miou, test_macc, test_acc, test_time))
        else:
            test_acc, test_miou, test_macc = 0, 0, 0
            print('-> Train accuracy: {:.3f}, \tLoss: {:.3f}, \tEpoch_time: {:.3f}'.format(acc, loss, epoch_time))

        stats.append({'epoch': epoch, 'loss': loss, 'oacc': acc, 'miou': miou, 'oacc_test': test_acc, 'test_miou': test_miou, 'test_macc': test_macc})

        if epoch % args.save_nth_epoch == 0 or epoch==args.epochs-1:
            with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                json.dump(stats, outfile)
            torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},
                       os.path.join(args.odir, 'epoch{}_model.pth.tar'.format(epoch)))

        if math.isnan(loss): break
        scheduler.step()


    if len(stats)>0:
        with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
            json.dump(stats, outfile)

    # Final evaluation
    if args.test_multisamp_n>0:
        acc_test, oacc_test, avg_iou_test, per_class_iou_test, predictions_test, avg_acc_test, confusion_matrix = eval_final()
        print('-> Multisample {}: Test accuracy: {}, \tTest oAcc: {}, \tTest avgIoU: {}, \tTest mAcc: {}'.format(args.test_multisamp_n, acc_test, oacc_test, avg_iou_test, avg_acc_test))
        with h5py.File(os.path.join(args.odir, 'predictions_'+args.db_test_name+'.h5'), 'w') as hf:
            for fname, o_cpu in predictions_test.items():
                hf.create_dataset(name=fname, data=o_cpu) #(0-based classes)
        with open(os.path.join(args.odir, 'scores_'+args.db_test_name+'.txt'), 'w') as outfile:
            json.dump([{'epoch': args.start_epoch, 'acc_test': acc_test, 'oacc_test': oacc_test, 'avg_iou_test': avg_iou_test, 'per_class_iou_test': per_class_iou_test, 'avg_acc_test': avg_acc_test}], outfile)
        np.save(os.path.join(args.odir, 'pointwise_cm.npy'), confusion_matrix)





  
def resume(args, dbinfo):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    
    checkpoint['args'].model_config = args.model_config #to ensure compatibility with previous arguments convention
    #this should be removed once new models are uploaded
    
    model = create_model(checkpoint['args'], dbinfo) #use original arguments, architecture can't change
    optimizer = create_optimizer(args, model)
    
    model.load_state_dict(checkpoint['state_dict'])
    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups: group['initial_lr'] = args.lr
    args.start_epoch = checkpoint['epoch']
    try:
        stats = json.loads(open(os.path.join(os.path.dirname(args.resume), 'trainlog.txt')).read())
    except:
        stats = []
    return model, optimizer, stats
    
def create_model(args, dbinfo):
    """ Creates model """
    model = nn.Module()

    nfeat = args.ptn_widths[1][-1]
    model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit)

    model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)

    model.att_lab = attention_ext_module.AttentionEXTModule(out_c=args.n_labels)
    model.att_ext = attention_ext_module.AttentionEXTModule(out_c=args.n_labels)
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)
    
    if args.cuda: 
        model.cuda()
    return model 

def create_optimizer(args, model):
    if args.optim=='sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim=='adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

def set_seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: 
        torch.cuda.manual_seed(seed)    

def filter_valid(output, target, other=None):
    """ Removes predictions for nodes without ground truth """
    idx = target!=-100
    if other is not None:
        return output[idx,:], target[idx], other[idx,...]
    return output[idx,:], target[idx]
    
def meter_value(meter):   
    return meter.value()[0] if meter.n>0 else 0


if __name__ == "__main__": 
    main()
