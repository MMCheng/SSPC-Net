# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict


def extension_accum2(input, score, fea, weak_label, edges, th=0.7, undirected=True, ext_max=80):
    # input: Nsp*c*n
    # score: Nsp*num_classes
    # fea: Nsp*c
    # weak_label: Nsp

    weak_label = torch.squeeze(weak_label) # Nsp

    score_soft = F.softmax(score, dim=-1) # Nsp*num_classes
    Nsp = input.shape[0]

    mask_label = (weak_label < 255).squeeze() # Nsp
    mask_unlabel = (weak_label == 255).squeeze() # Nsp
    labeled_idx = torch.nonzero(mask_label).squeeze().long() # Nsp_unlabel
    unlabel_idx = torch.nonzero(mask_unlabel).squeeze().long() # Nsp_unlabel

    candidates = defaultdict(list)
    for i in range(edges.shape[0]):
        x = edges[i, 0].item()
        y = edges[i, 1].item()
        if x != y:
            candidates[x].append(y)
            if undirected:
                candidates[y].append(x)


    weak_label2 = []
    score2 = []
    extend_idx = []
    for i in range(labeled_idx.shape[0]):
        sp_idx = labeled_idx[i] # idx of the labeled sp
        sp_wl = weak_label[sp_idx]

        neighbors = candidates[sp_idx.item()] 
        neighbor_score = -1
        neighbor_ext = -1
        for nei in neighbors:
            if weak_label[nei] == 255:
                score_vec_point = score_soft[nei] # 13
                s, p = torch.max(score_vec_point, 0)
                if (p==sp_wl) & (s>neighbor_score):
                    neighbor_ext = nei 
                    neighbor_score = s

        if (neighbor_score>th) & (neighbor_ext not in extend_idx) & (neighbor_ext not in labeled_idx):
            weak_label2.append(sp_wl.unsqueeze(0))
            score2.append(score[neighbor_ext, :].unsqueeze(0))
            extend_idx.append(neighbor_ext)

    if len(extend_idx)>0:
        weak_label2 = torch.cat(weak_label2)
        score2 = torch.cat(score2, 0)
    extend_idx = torch.tensor(extend_idx)


    if extend_idx.shape[0]>ext_max:
        pred_v, _ = torch.max(score2, 1)
        _, indices = torch.sort(pred_v, 0, descending=True)

        weak_label2_sample = weak_label2[indices[:ext_max]]
        score2_sample = score2[indices[:ext_max], :]
        extend_idx_sample = extend_idx[indices[:ext_max]]
    else:
        weak_label2_sample = weak_label2
        score2_sample = score2
        extend_idx_sample = extend_idx

    score1 = score[mask_label, :]
    weak_label1 = weak_label[mask_label]
    
    return score1, weak_label1, score2_sample, weak_label2_sample, extend_idx_sample, labeled_idx
