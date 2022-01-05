# encoding=utf-8

from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import math
import transforms3d
import torch
import ecc
import h5py
from sklearn import preprocessing
import igraph
from math import floor



def spg_edge_features(edges, node_att, edge_att, args):
    """ Assembles edge features from edge attributes and differences of node attributes. """
    columns = []
    for attrib in args.edge_attribs.split(','):
        attrib = attrib.split('/')
        a, opt = attrib[0], attrib[1].lower() if len(attrib)==2 else ''

        if a in ['delta_avg', 'delta_std']:
            columns.append(edge_att[a])
        elif a=='constant': # for isotropic baseline
            columns.append(np.ones((edges.shape[0],1), dtype=np.float32))
        elif a in ['nlength','surface','volume', 'size', 'xyz']:
            attr = node_att[a]
            if opt=='d': # difference
                attr = attr[edges[:,0],:] - attr[edges[:,1],:]
            elif opt=='ld': # log ratio
                attr = np.log(attr + 1e-10)
                attr = attr[edges[:,0],:] - attr[edges[:,1],:]
            elif opt=='r': # ratio
                attr = attr[edges[:,0],:] / (attr[edges[:,1],:] + 1e-10)
            else:
                raise NotImplementedError
            columns.append(attr)
        else:
            raise NotImplementedError

    return np.concatenate(columns, axis=1).astype(np.float32)

def scaler01(trainlist, testlist, transform_train=True):
    """ Scale edge features to 0 mean 1 stddev """
    edge_feats = np.concatenate([ trainlist[i][3] for i in range(len(trainlist)) ], 0) # (Nedge_1 + Nedge_2 + ... + Nedge_n)*13
    scaler = preprocessing.StandardScaler().fit(edge_feats)

    if transform_train:
        for i in range(len(trainlist)):
            scaler.transform(trainlist[i][3], copy=False)
            # default copy=True
            # if copy=True, the edge features will remain the same and not be replaced with normalized features
    for i in range(len(testlist)):
        scaler.transform(testlist[i][3], copy=False)
    return trainlist, testlist

def spg_reader(args, fname, incl_dir_in_name=False):
    """ Loads a supergraph from H5 file. """
    f = h5py.File(fname,'r')

    if f['sp_labels'].size > 0:

        node_gt_size = f['sp_labels'][:].astype(np.int64) # Nsp*13, column 0: number of unlabeled points
        node_gt = np.argmax(node_gt_size, 1)[:,None] # (Nsp, ) --> (Nsp, 1), find the certain sp label

        room_weak_labels = f['sp_weak_labels'][:].astype(np.int64) # Nsp*14, column 0: number of unlabeled points

        num_sp = room_weak_labels.shape[0]
        node_weak_label = np.argmax(room_weak_labels[:, :args.n_labels], 1)[:, None] # (Nsp, ) --> (Nsp, 1), find the certain sp weak label
        node_weak_label[room_weak_labels[:,:args.n_labels].sum(1)==0,:] = 255   # (Nsp, 1), superpoints without weak labels are ignored in loss computation


        for i in range(num_sp):
            if node_weak_label[i, 0] < 255:
                sp_weak_labels =  room_weak_labels[i, :] # (14,)
                mask = sp_weak_labels != 0   # (14,)    0~12, >255

                if np.sum(mask) <= 2:
                    weak_label = np.argmax(sp_weak_labels[:args.n_labels]) # find the certain sp weak label
                    if sp_weak_labels[:args.n_labels].sum()==0:
                        weak_label = 255   # (Nsp, 1), superpoints without weak labels are ignored in loss computation
                    node_weak_label[i, 0] = weak_label

                else:
                    num_labels_in_sp = int(np.sum(mask)-1) # number of valid weak labels
                    idx_labels = np.nonzero(sp_weak_labels)[0] # find the weak label
                    several_labels = idx_labels[:num_labels_in_sp]  # revove the ignore label in the last place

                    corres_p_count = node_gt_size[i, several_labels]
                    choose_label = several_labels[np.argmax(corres_p_count, 0)]
                    node_weak_label[i, 0] = choose_label
    else:
        N = f['sp_point_count'].shape[0]
        node_gt_size = np.concatenate([f['sp_point_count'][:].astype(np.int64), np.zeros((N,8), dtype=np.int64)], 1)
        node_gt = np.zeros((N,1), dtype=np.int64)

    node_att = {}
    node_att['xyz'] = f['sp_centroids'][:]
    node_att['nlength'] = np.maximum(0, f['sp_length'][:])
    node_att['volume'] = np.maximum(0, f['sp_volume'][:] ** 2)
    node_att['surface'] = np.maximum(0, f['sp_surface'][:] ** 2)
    node_att['size'] = f['sp_point_count'][:]

    edges = np.concatenate([ f['source'][:], f['target'][:] ], axis=1).astype(np.int64)

    edge_att = {}
    edge_att['delta_avg'] = f['se_delta_mean'][:]
    edge_att['delta_std'] = f['se_delta_std'][:]

    if args.spg_superedge_cutoff > 0:
        filtered = np.linalg.norm(edge_att['delta_avg'],axis=1) < args.spg_superedge_cutoff
        edges = edges[filtered,:]
        edge_att['delta_avg'] = edge_att['delta_avg'][filtered,:]
        edge_att['delta_std'] = edge_att['delta_std'][filtered,:]

    edge_feats = spg_edge_features(edges, node_att, edge_att, args)


    name = os.path.basename(fname)[:-len('.h5')]

    if not os.path.exists(os.path.join(args.extension_dir, 'epoch_0', '{}.txt'.format(name))):
        if not os.path.exists(os.path.join(args.extension_dir, 'epoch_0')):
            os.makedirs(os.path.join(args.extension_dir, 'epoch_0'))
        extension_full = np.ones((num_sp,))*(-1)
        np.savetxt(os.path.join(args.extension_dir, 'epoch_0', '{}.txt'.format(name)), extension_full, fmt='%d')
    return node_gt, node_gt_size, node_weak_label, edges, edge_feats, name



def spg_to_igraph(node_gt, node_gt_size, node_weak_label, edges, edge_feats, fname):
    """ Builds representation of superpoint graph as igraph. """
    targets = np.concatenate([node_weak_label, node_gt, node_gt_size], axis=1) # Nsp*16
    G = igraph.Graph(n=node_gt.shape[0], edges=edges.tolist(), directed=True, # n: number of sp, edges: edges' vertex
                     edge_attrs={'f':edge_feats},                               # edge features
                     vertex_attrs={'v':list(range(node_gt.shape[0])), 't':targets, 's':node_gt_size.sum(1)}) # v: sp_id, sp_labels, sp points count
    return G, fname

def random_neighborhoods(G, num, order):
    """ Samples `num` random neighborhoods of size `order`.
        Graph nodes are then treated as set, i.e. after hardcutoff, neighborhoods may be broken (sort of data augmentation). """
    
    # 1. randomly select k superpoints and find the i-th orderneighbors of these superpoints
    centers = random.sample(range(G.vcount()), k=num)
    neighb = G.neighborhood(centers, order)

    # 2. build a set of all these center superpoints and neighboring superpoints
    subset = [item for sublist in neighb for item in sublist]
    subset = set(subset)

    mask = np.array(G.vs['t'])[:, 0] < 255 # (Nsp, ) sp weak label
    subset_valid = np.nonzero(mask)[0] # sp with valid weak labels should be remained
    subset = sorted(subset | set(subset_valid))

    # 3. return a new graph consists of the new set
    return G.subgraph(subset)

def k_big_enough(G, minpts, k):
    """ Returns a induced graph on maximum k superpoints of size >= minpts (smaller ones are not counted) """
    valid = np.array(G.vs['s']) >= minpts
    n = np.argwhere(np.cumsum(valid)<=k)[-1][0]+1

    subset = set(range(n))
    mask = np.array(G.vs['t'])[:, 0] < 255 # (Nsp, ) sp weak label
    subset_valid = np.nonzero(mask)[0] # sp with valid weak labels should be remained
    subset = sorted(subset | set(subset_valid))
    return G.subgraph(range(n)) # return the graph containing the first n points


def loader(entry, train, args, db_path, test_seed_offset=0):
    """ Prepares a superpoint graph (potentially subsampled in training) and associated superpoints. """
    G, fname = entry # fname: Area_6_office_21
    if train:
        path = os.path.join(args.extension_dir, 'epoch_{:d}'.format(int(args.ext_epoch//args.ext_epoch_gap)), '{}.txt'.format(fname)) # results/s3dis/ext_dir/cv5/extension_log/Area_6_office_25.txt
        extension_full = np.loadtxt(path)
        extension_full = extension_full.astype(np.int32) # (N, ),  whether the sp was extended

    # 1) subset (neighborhood) selection of (permuted) superpoint graph
    if train:
        if 0 < args.spg_augm_hardcutoff < G.vcount(): # permute vertexes
            perm = list(range(G.vcount())); random.shuffle(perm)
            G = G.permute_vertices(perm)

        if 0 < args.spg_augm_nneigh < G.vcount(): # ignore the superpoints disconnected with others
            G = random_neighborhoods(G, args.spg_augm_nneigh, args.spg_augm_order)

        if 0 < args.spg_augm_hardcutoff < G.vcount():
            G = k_big_enough(G, args.ptn_minpts, args.spg_augm_hardcutoff)

    # 2) loading clouds for chosen superpoint graph nodes
    clouds_meta, clouds_flag = [], [] # meta: textual id of the superpoint; flag: 0/-1 if no cloud because too small
    clouds, clouds_global = [], [] # clouds: point cloud arrays; clouds_global: diameters before scaling

    current_num_sp = G.vcount()
    extension_sub = np.ones(shape=(current_num_sp, 2), dtype=np.int32)*(-255) # orig_sp_idx, extension_label


    for s in range(G.vcount()):

        if args.data_mode == 'voxel':
            cloud, diam, sp_flag = load_superpoint(args, db_path + '/sp_voxel_pc/' + fname + '.h5', G.vs[s]['v'], train, test_seed_offset)
        else:
            cloud, diam, sp_flag = load_superpoint(args, db_path + '/sp_point_pc/' + fname + '.h5', G.vs[s]['v'], train, test_seed_offset)


        clouds_meta.append('{}.{:d}'.format(fname,G.vs[s]['v'])) # str list: fname & original sp id, e.g. Area_4/hallway_3.1020
        clouds.append(cloud.T)      # array list: 14*128, pc of each sp
        clouds_global.append(diam)  # diameter of each sp's orginal pc
        if sp_flag:
            clouds_flag.append(0)       # int list: indicate whether this sp return points for ptn, 0: valid sp, -1: invalid sp
        else:
            clouds_flag.append(-1)       # int list: indicate whether this sp return points for ptn, 0: valid sp, -1: invalid sp

        if train:
            extension_sub[s, 0] = G.vs[s]['v']
            extension_sub[s, 1] = extension_full[G.vs[s]['v']]

    clouds_flag = np.array(clouds_flag) # (Nsp, )
    clouds = np.stack(clouds) # Nsp*14*128
    clouds_global = np.concatenate(clouds_global) # (Nsp, )

    # np.array(G.vs['t']):      Nsp*16
    # G: G.vcount():            Nsp
    # clouds_meta: list         len(clouds_meta): Nsp
    # clouds_flag:              (Nsp, )
    # clouds:                   (Nsp_valid, 14, 128)
    # clouds_global:            (Nsp_valid, )
    # extension_sub:            Nsp*2, [original_sp_id, extension_flag]
    # extension_full:           (N, ), extension flag
    if train:
        return np.array(G.vs['t']), G, clouds_meta, clouds_flag, clouds, clouds_global, fname, extension_sub, extension_full
    else:
        return np.array(G.vs['t']), G, clouds_meta, clouds_flag, clouds, clouds_global, fname
    



def cloud_edge_feats(edgeattrs):
    edgefeats = np.asarray(edgeattrs['f'])
    return torch.from_numpy(edgefeats), None

def eccpc_collate(batch):
    """ Collates a list of dataset samples into a single batch (adapted in ecc.graph_info_collate_classification())
    """
    targets, graphs, clouds_meta, clouds_flag, clouds, clouds_global, fnames, extension_subs, extension_fulls = list(zip(*batch))
    

    targets = torch.cat([torch.from_numpy(t) for t in targets], 0).long()
    GIs = [ecc.GraphConvInfo(graphs, cloud_edge_feats)]
    edges_for_ext = GIs[0].edges_for_ext

    if len(clouds_meta[0])>0:
        clouds = torch.cat([torch.from_numpy(f) for f in clouds], 0)
        clouds_global = torch.cat([torch.from_numpy(f) for f in clouds_global], 0)
        clouds_flag = torch.cat([torch.from_numpy(f) for f in clouds_flag], 0)
        clouds_meta = [item for sublist in clouds_meta for item in sublist]
    clouds_orig = clouds
    mask = clouds_flag == 0
    clouds = clouds[mask, :, :]
    clouds_global = clouds_global[mask]
    
    num_sp_list = [f.shape[0] for f in extension_subs]
    ext_mask = torch.cat([torch.unsqueeze(torch.from_numpy(t[:, 1]), -1) for t in extension_subs], 0).long().squeeze(-1)
    
    return targets, GIs, (clouds_meta, clouds_flag, clouds, clouds_global), clouds_orig, edges_for_ext, fnames, (ext_mask, extension_subs, extension_fulls), num_sp_list
    # ext_mask:                 (Nsp*batches, ), extension flag, -1/1
    # extension_subs:           list --> Nsp*2, [original_sp_id, extension_flag]
    # extension_fulls:          list --> (N, ), extension flag


def eccpc_collate_test(batch):
    """ Collates a list of dataset samples into a single batch (adapted in ecc.graph_info_collate_classification())
    """
    targets, graphs, clouds_meta, clouds_flag, clouds, clouds_global, fnames = list(zip(*batch))
    
    targets = torch.cat([torch.from_numpy(t) for t in targets], 0).long()
    GIs = [ecc.GraphConvInfo(graphs, cloud_edge_feats)]
    edges_for_ext = GIs[0].edges_for_ext

    if len(clouds_meta[0])>0:
        clouds = torch.cat([torch.from_numpy(f) for f in clouds], 0)
        clouds_global = torch.cat([torch.from_numpy(f) for f in clouds_global], 0)
        clouds_flag = torch.cat([torch.from_numpy(f) for f in clouds_flag], 0)
        clouds_meta = [item for sublist in clouds_meta for item in sublist]
    clouds_orig = clouds
    mask = clouds_flag == 0
    clouds = clouds[mask, :, :]
    clouds_global = clouds_global[mask]
    
    return targets, GIs, (clouds_meta, clouds_flag, clouds, clouds_global), clouds_orig, edges_for_ext, fnames



############### POINT CLOUD PROCESSING ##########

def load_superpoint(args, fname, id, train, test_seed_offset):
    """ """
    hf = h5py.File(fname,'r')
    P = hf['{:d}_data'.format(id)] # N*14
    N = P.shape[0]
    if N < args.ptn_minpts: # skip if too few pts (this must be consistent at train and test time)
        sp_flag = False
    else:
        sp_flag = True
    P = P[:].astype(np.float32) # N*14

    rs = np.random.random.__self__ if train else np.random.RandomState(seed=id+test_seed_offset) # fix seed for test

    if N > args.ptn_npts: # need to subsample
        ii = rs.choice(N, args.ptn_npts)
        P = P[ii, ...]
    elif N < args.ptn_npts: # need to pad by duplication
        ii = rs.choice(N, args.ptn_npts - N)
        P = np.concatenate([P, P[ii,...]], 0)

    if args.pc_xyznormalize: # points inner sp normalized into a unit ball
        # normalize xyz into unit ball, i.e. in [-0.5,0.5]
        diameter = np.max(np.max(P[:,:3],axis=0) - np.min(P[:,:3],axis=0))
        P[:,:3] = (P[:,:3] - np.mean(P[:,:3], axis=0, keepdims=True)) / (diameter + 1e-10)
    else:
        diameter = 0.0
        P[:,:3] = (P[:,:3] - np.mean(P[:,:3], axis=0, keepdims=True))

    if args.pc_attribs != '':
        columns = []
        if 'xyz' in args.pc_attribs: columns.append(P[:,:3])
        if 'rgb' in args.pc_attribs: columns.append(P[:,3:6])
        if 'e' in args.pc_attribs: columns.append(P[:,6,None])
        if 'lpsv' in args.pc_attribs: columns.append(P[:,7:11])
        if 'XYZ' in args.pc_attribs: columns.append(P[:,11:14])
        P = np.concatenate(columns, axis=1)

    if train:
        P = augment_cloud(P, args)

    return P, np.array([diameter], dtype=np.float32), sp_flag # (128, 14), (1, )


def augment_cloud(P, args):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1/args.pc_augm_scale, args.pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot: # rotate around the z-axis
        angle = random.uniform(0, 2*math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # z=upright assumption
    if args.pc_augm_mirror_prob > 0: # mirroring x&y, not z
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
    P[:,:3] = np.dot(P[:,:3], M.T)

    if args.pc_augm_jitter: # jitter
        sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
    return P

