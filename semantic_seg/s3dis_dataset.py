# encoding=utf-8

from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg

def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    testlist, trainlist = [], []

    if args.data_mode == 'voxel':
        path = os.path.join(args.S3DIS_PATH, 'graph_v')
    else:
        path = os.path.join(args.S3DIS_PATH, 'graph_p')


    for fname in sorted(os.listdir(path)):
        area = int(fname[5])
        if fname.endswith(".h5"):
            if area != args.cvfold:
                trainlist.append(spg.spg_reader(args, os.path.join(path, fname), True))
            elif area == args.cvfold:
                testlist.append(spg.spg_reader(args, os.path.join(path, fname), True))

   


    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.S3DIS_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.S3DIS_PATH, test_seed_offset=test_seed_offset))




def get_datasets_test(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    testlist, trainlist = [], []

    if args.data_mode == 'voxel':
        path = os.path.join(args.S3DIS_PATH, 'graph_v')
    else:
        path = os.path.join(args.S3DIS_PATH, 'graph_p')


    for fname in sorted(os.listdir(path)):
        area = int(fname[5])
        if fname.endswith(".h5"):
            if area != args.cvfold:
                trainlist.append(spg.spg_reader(args, os.path.join(path, fname), True))
            if area == args.cvfold:
                testlist.append(spg.spg_reader(args, os.path.join(path, fname), True))


    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.S3DIS_PATH, test_seed_offset=test_seed_offset))
    


def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 14 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 13,
        'inv_class_map': {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window', 6:'door', 7:'chair', 8:'table', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'},

    }


