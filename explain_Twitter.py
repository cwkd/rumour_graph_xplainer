import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torch_sparse
# from torch_scatter.utils import broadcast
from torch_scatter import scatter, scatter_mean
# import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.nn.conv import gcn_conv

from model.Twitter.BiGCN_Twitter import BiGCN
from model.Twitter.EBGCN import EBGCN
from Process.process import loadBiData, loadTree
from Process.rand5fold import load5foldData

# from torch_geometric.utils import add_remaining_self_loops
# from torch_geometric.utils.num_nodes import maybe_num_nodes

import lrp_pytorch.modules.utils as lrp_utils
from lrp_pytorch.modules.base import safe_divide
from tqdm import tqdm
import copy
import argparse
import json

FOLD_2_EVENTNAME = {0: 'charliehebdo',
                    1: 'ebola',
                    2: 'ferguson',
                    3: 'germanwings',
                    4: 'gurlitt',
                    5: 'ottawashooting',
                    6: 'prince',
                    7: 'putinmissing',
                    8: 'sydneysiege'}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPLAIN_DIR = os.path.join(DATA_DIR, 'explain')
CENTRALITY_DIR = os.path.join(DATA_DIR, 'centrality')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'model', 'Twitter', 'checkpoints')
random.seed(0)

LAYERS = ['input', 'td_conv1', 'td_conv2', 'bu_conv2', 'bu_conv2']

LRP_PARAMS = {
    'linear_eps': 1e-6,
    'gcn_conv': 1e-6,
    'bigcn': 1e-6,
    'ebgcn': 1e-6,
    'mode': 'lrp'
}


def swap_elements(lst, idx1, idx2):
    if isinstance(idx1, int):
        lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
        return lst
    for i, j in zip(idx1, idx2):
        lst[i], lst[j] = lst[j], lst[i]
    return lst


def compute_gcl_relevance(text_vecs, edge_index, edge_weights, relevance_map):
    text_vecs = np.asarray(text_vecs)
    edge_index = np.asarray(edge_index)
    edge_weights = np.asarray(edge_weights)
    relevance_map = np.asarray(relevance_map)
    # print(text_vecs.shape, edge_index.shape, edge_weights.shape, relevance_map.shape)
    edge_relevance = np.zeros((edge_index.shape[1], text_vecs.shape[1]))
    node_count = text_vecs.shape[0]
    non_self_edge_count = edge_index.shape[1] - node_count
    z_map = np.zeros(text_vecs.shape)

    # forward pass; obtain z
    for edge_num in range(edge_index.shape[-1]):
        src, dst = edge_index[0, edge_num], edge_index[1, edge_num]
        z_map[dst] += text_vecs[src] * edge_weights[edge_num]
    try:
        assert relevance_map.shape == z_map.shape
    except:
        print(relevance_map.shape, z_map.shape)
    assert relevance_map.shape == z_map.shape
    s_map = safe_divide(torch.tensor(relevance_map), torch.tensor(z_map), 1e-6, 1e-6).numpy()
    # backward pass; obtain c
    non_self_edge_index = np.zeros((2, non_self_edge_count))
    for edge_num in range(edge_index.shape[-1]):
        src, dst = edge_index[0, edge_num], edge_index[1, edge_num]
        if src != dst:
            non_self_edge_index[0, edge_num] = src
            non_self_edge_index[1, edge_num] = dst
        edge_relevance[edge_num] += s_map[dst] * edge_weights[edge_num] * text_vecs[src]
    # print(edge_relevance)
    node_relevance = edge_relevance[non_self_edge_count:]
    edge_relevance = edge_relevance[:non_self_edge_count]
    assert edge_relevance.shape[0] == non_self_edge_count
    assert node_relevance.shape[0] == node_count
    return node_relevance, edge_relevance, non_self_edge_index


def compute_text_contribution(text_act1, text_act2):
    return list(map(lambda x, y: x + y, text_act1, text_act2))


def get_text_ranked_list(conv1_text_relevance, conv2_text_relevance):
    text_relevance = compute_text_contribution(
        text_act1=conv1_text_relevance.sum(-1),
        text_act2=conv2_text_relevance.sum(-1))
    text_ranking = torch.topk(torch.as_tensor(text_relevance), k=len(text_relevance))
    return [text_ranking.indices.tolist(), text_ranking.values.tolist()], text_relevance


def get_edge_ranked_list(conv1_edge_relevance, conv2_edge_relevance):
    edge_relevance = conv1_edge_relevance.sum(-1) + conv2_edge_relevance.sum(-1)
    edge_ranking = torch.topk(torch.as_tensor(edge_relevance), k=len(edge_relevance))
    return [edge_ranking.indices.tolist(), edge_ranking.values.tolist()], edge_relevance


def process_lrp_contributions(ref_dict):
    # TD
    bigcn_td_conv1_text = ref_dict['td_gcn_explanations']['conv1_text'][0]
    bigcn_td_conv2_text = ref_dict['td_gcn_explanations']['conv2_text'][0]
    bigcn_td_conv1_edge_weights = ref_dict['td_gcn_explanations']['conv1_edge_weights']
    bigcn_td_conv2_edge_weights = ref_dict['td_gcn_explanations']['conv2_edge_weights']

    bigcn_lrp_class0_td_conv1 = ref_dict['lrp_class0_td_conv1']
    bigcn_lrp_class1_td_conv1 = ref_dict['lrp_class1_td_conv1']
    bigcn_lrp_class2_td_conv1 = ref_dict['lrp_class2_td_conv1']
    bigcn_lrp_class3_td_conv1 = ref_dict['lrp_class3_td_conv1']

    bigcn_lrp_class0_td_conv2 = ref_dict['lrp_class0_td_conv2']
    bigcn_lrp_class1_td_conv2 = ref_dict['lrp_class1_td_conv2']
    bigcn_lrp_class2_td_conv2 = ref_dict['lrp_class2_td_conv2']
    bigcn_lrp_class3_td_conv2 = ref_dict['lrp_class3_td_conv2']

    logits = ref_dict['logits']
    logit_weights = np.exp(np.asarray(logits))[0]

    # Conv 1
    bigcn_lrp_class0_td_conv1_text, bigcn_lrp_class0_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_lrp_class0_td_conv1)
    bigcn_lrp_class1_td_conv1_text, bigcn_lrp_class1_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_lrp_class1_td_conv1)
    bigcn_lrp_class2_td_conv1_text, bigcn_lrp_class2_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_lrp_class2_td_conv1)
    bigcn_lrp_class3_td_conv1_text, bigcn_lrp_class3_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_lrp_class3_td_conv1)

    # Conv 2
    bigcn_lrp_class0_td_conv2_text, bigcn_lrp_class0_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_lrp_class0_td_conv2)
    bigcn_lrp_class1_td_conv2_text, bigcn_lrp_class1_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_lrp_class1_td_conv2)
    bigcn_lrp_class2_td_conv2_text, bigcn_lrp_class2_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_lrp_class2_td_conv2)
    bigcn_lrp_class3_td_conv2_text, bigcn_lrp_class3_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_lrp_class3_td_conv2)

    bigcn_lrp_class0_td_text, t0 = get_text_ranked_list(bigcn_lrp_class0_td_conv1_text,
                                                        bigcn_lrp_class0_td_conv2_text)
    bigcn_lrp_class1_td_text, t1 = get_text_ranked_list(bigcn_lrp_class1_td_conv1_text,
                                                        bigcn_lrp_class1_td_conv2_text)
    bigcn_lrp_class2_td_text, t2 = get_text_ranked_list(bigcn_lrp_class1_td_conv1_text,
                                                        bigcn_lrp_class2_td_conv2_text)
    bigcn_lrp_class3_td_text, t3 = get_text_ranked_list(bigcn_lrp_class1_td_conv1_text,
                                                        bigcn_lrp_class3_td_conv2_text)

    bigcn_lrp_allclass_td_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                 np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    bigcn_lrp_allclass_td_text /= bigcn_lrp_allclass_td_text.sum()
    bigcn_lrp_allclass_td_text = torch.topk(torch.as_tensor(bigcn_lrp_allclass_td_text),
                                            k=bigcn_lrp_allclass_td_text.shape[0])
    bigcn_lrp_allclass_td_text = [bigcn_lrp_allclass_td_text.indices.tolist(),
                                  bigcn_lrp_allclass_td_text.values.tolist()]

    bigcn_lrp_class0_td_edge, e0 = get_edge_ranked_list(bigcn_lrp_class0_td_conv1_edge,
                                                        bigcn_lrp_class0_td_conv2_edge)
    bigcn_lrp_class1_td_edge, e1 = get_edge_ranked_list(bigcn_lrp_class1_td_conv1_edge,
                                                        bigcn_lrp_class1_td_conv2_edge)
    bigcn_lrp_class2_td_edge, e2 = get_edge_ranked_list(bigcn_lrp_class1_td_conv1_edge,
                                                        bigcn_lrp_class2_td_conv2_edge)
    bigcn_lrp_class3_td_edge, e3 = get_edge_ranked_list(bigcn_lrp_class1_td_conv1_edge,
                                                        bigcn_lrp_class3_td_conv2_edge)
    bigcn_lrp_allclass_td_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                 np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    bigcn_lrp_allclass_td_edge /= bigcn_lrp_allclass_td_edge.sum()
    bigcn_lrp_allclass_td_edge = torch.topk(torch.as_tensor(bigcn_lrp_allclass_td_edge),
                                            k=bigcn_lrp_allclass_td_edge.shape[0])
    bigcn_lrp_allclass_td_edge = [bigcn_lrp_allclass_td_edge.indices.tolist(),
                                  bigcn_lrp_allclass_td_edge.values.tolist()]
    # BU
    bigcn_bu_conv1_text = ref_dict['bu_gcn_explanations']['conv1_text'][0]
    bigcn_bu_conv2_text = ref_dict['bu_gcn_explanations']['conv2_text'][0]
    bigcn_bu_conv1_edge_weights = ref_dict['bu_gcn_explanations']['conv1_edge_weights']
    bigcn_bu_conv2_edge_weights = ref_dict['bu_gcn_explanations']['conv2_edge_weights']

    bigcn_lrp_class0_bu_conv1 = ref_dict['lrp_class0_bu_conv1']
    bigcn_lrp_class1_bu_conv1 = ref_dict['lrp_class1_bu_conv1']
    bigcn_lrp_class2_bu_conv1 = ref_dict['lrp_class2_bu_conv1']
    bigcn_lrp_class3_bu_conv1 = ref_dict['lrp_class3_bu_conv1']

    bigcn_lrp_class0_bu_conv2 = ref_dict['lrp_class0_bu_conv2']
    bigcn_lrp_class1_bu_conv2 = ref_dict['lrp_class1_bu_conv2']
    bigcn_lrp_class2_bu_conv2 = ref_dict['lrp_class2_bu_conv2']
    bigcn_lrp_class3_bu_conv2 = ref_dict['lrp_class3_bu_conv2']

    logits = ref_dict['logits']
    logit_weights = np.exp(np.asarray(logits))[0]

    # Conv 1
    bigcn_lrp_class0_bu_conv1_text, bigcn_lrp_class0_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_lrp_class0_bu_conv1)
    bigcn_lrp_class1_bu_conv1_text, bigcn_lrp_class1_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_lrp_class1_bu_conv1)
    bigcn_lrp_class2_bu_conv1_text, bigcn_lrp_class2_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_lrp_class2_bu_conv1)
    bigcn_lrp_class3_bu_conv1_text, bigcn_lrp_class3_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_lrp_class3_bu_conv1)

    # Conv 2
    bigcn_lrp_class0_bu_conv2_text, bigcn_lrp_class0_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_lrp_class0_bu_conv2)
    bigcn_lrp_class1_bu_conv2_text, bigcn_lrp_class1_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_lrp_class1_bu_conv2)
    bigcn_lrp_class2_bu_conv2_text, bigcn_lrp_class2_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_lrp_class2_bu_conv2)
    bigcn_lrp_class3_bu_conv2_text, bigcn_lrp_class3_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_lrp_class3_bu_conv2)

    bigcn_lrp_class0_bu_text, t0 = get_text_ranked_list(bigcn_lrp_class0_bu_conv1_text,
                                                        bigcn_lrp_class0_bu_conv2_text)
    bigcn_lrp_class1_bu_text, t1 = get_text_ranked_list(bigcn_lrp_class1_bu_conv1_text,
                                                        bigcn_lrp_class1_bu_conv2_text)
    bigcn_lrp_class2_bu_text, t2 = get_text_ranked_list(bigcn_lrp_class1_bu_conv1_text,
                                                        bigcn_lrp_class2_bu_conv2_text)
    bigcn_lrp_class3_bu_text, t3 = get_text_ranked_list(bigcn_lrp_class1_bu_conv1_text,
                                                        bigcn_lrp_class3_bu_conv2_text)
    bigcn_lrp_allclass_bu_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                 np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    bigcn_lrp_allclass_bu_text /= bigcn_lrp_allclass_bu_text.sum()
    bigcn_lrp_allclass_bu_text = torch.topk(torch.as_tensor(bigcn_lrp_allclass_bu_text),
                                            k=bigcn_lrp_allclass_bu_text.shape[0])
    bigcn_lrp_allclass_bu_text = [bigcn_lrp_allclass_bu_text.indices.tolist(),
                                  bigcn_lrp_allclass_bu_text.values.tolist()]

    bigcn_lrp_class0_bu_edge, e0 = get_edge_ranked_list(bigcn_lrp_class0_bu_conv1_edge,
                                                        bigcn_lrp_class0_bu_conv2_edge)
    bigcn_lrp_class1_bu_edge, e1 = get_edge_ranked_list(bigcn_lrp_class1_bu_conv1_edge,
                                                        bigcn_lrp_class1_bu_conv2_edge)
    bigcn_lrp_class2_bu_edge, e2 = get_edge_ranked_list(bigcn_lrp_class1_bu_conv1_edge,
                                                        bigcn_lrp_class2_bu_conv2_edge)
    bigcn_lrp_class3_bu_edge, e3 = get_edge_ranked_list(bigcn_lrp_class1_bu_conv1_edge,
                                                        bigcn_lrp_class3_bu_conv2_edge)
    bigcn_lrp_allclass_bu_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                 np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    bigcn_lrp_allclass_bu_edge /= bigcn_lrp_allclass_bu_edge.sum()
    bigcn_lrp_allclass_bu_edge = torch.topk(torch.as_tensor(bigcn_lrp_allclass_bu_edge),
                                            k=bigcn_lrp_allclass_bu_edge.shape[0])
    bigcn_lrp_allclass_bu_edge = [bigcn_lrp_allclass_bu_edge.indices.tolist(),
                                  bigcn_lrp_allclass_bu_edge.values.tolist()]

    # Whole Model (Keep)
    # bigcn_lrp_class0 = ref_dict['lrp_class0_top_k']
    # bigcn_lrp_class1 = ref_dict['lrp_class1_top_k']
    # bigcn_lrp_class2 = ref_dict['lrp_class2_top_k']
    # bigcn_lrp_class3 = ref_dict['lrp_class3_top_k']
    # bigcn_lrp_allclass = ref_dict['lrp_allclass_top_k']

    # Extra
    bigcn_nr_fc = np.asarray(ref_dict['lrp_class0_fc'])
    bigcn_fr_fc = np.asarray(ref_dict['lrp_class1_fc'])
    bigcn_tr_fc = np.asarray(ref_dict['lrp_class2_fc'])
    bigcn_ur_fc = np.asarray(ref_dict['lrp_class3_fc'])

    bigcn_nr_fc = [bigcn_nr_fc[:bigcn_nr_fc.shape[-1] // 2].sum(),
                   bigcn_nr_fc[bigcn_nr_fc.shape[-1] // 2:].sum()]
    bigcn_fr_fc = [bigcn_fr_fc[:bigcn_fr_fc.shape[-1] // 2].sum(),
                   bigcn_fr_fc[bigcn_fr_fc.shape[-1] // 2:].sum()]
    bigcn_tr_fc = [bigcn_tr_fc[:bigcn_tr_fc.shape[-1] // 2].sum(),
                   bigcn_tr_fc[bigcn_tr_fc.shape[-1] // 2:].sum()]
    bigcn_ur_fc = [bigcn_ur_fc[:bigcn_ur_fc.shape[-1] // 2].sum(),
                   bigcn_ur_fc[bigcn_ur_fc.shape[-1] // 2:].sum()]

    bigcn_nr_td_fc = np.asarray(ref_dict['lrp_class0_td_conv1_fc'])
    bigcn_fr_td_fc = np.asarray(ref_dict['lrp_class1_td_conv1_fc'])
    bigcn_tr_td_fc = np.asarray(ref_dict['lrp_class2_td_conv1_fc'])
    bigcn_ur_td_fc = np.asarray(ref_dict['lrp_class3_td_conv1_fc'])
    bigcn_nr_td_fc = [list(range(bigcn_nr_td_fc.shape[0])), bigcn_nr_td_fc.sum(-1).tolist()]
    bigcn_fr_td_fc = [list(range(bigcn_fr_td_fc.shape[0])), bigcn_fr_td_fc.sum(-1).tolist()]
    bigcn_tr_td_fc = [list(range(bigcn_tr_td_fc.shape[0])), bigcn_tr_td_fc.sum(-1).tolist()]
    bigcn_ur_td_fc = [list(range(bigcn_ur_td_fc.shape[0])), bigcn_ur_td_fc.sum(-1).tolist()]

    bigcn_nr_bu_fc = np.asarray(ref_dict['lrp_class0_bu_conv1_fc'])
    bigcn_fr_bu_fc = np.asarray(ref_dict['lrp_class1_bu_conv1_fc'])
    bigcn_tr_bu_fc = np.asarray(ref_dict['lrp_class2_bu_conv1_fc'])
    bigcn_ur_bu_fc = np.asarray(ref_dict['lrp_class3_bu_conv1_fc'])
    bigcn_nr_bu_fc = [list(range(bigcn_nr_bu_fc.shape[0])), bigcn_nr_bu_fc.sum(-1).tolist()]
    bigcn_fr_bu_fc = [list(range(bigcn_fr_bu_fc.shape[0])), bigcn_fr_bu_fc.sum(-1).tolist()]
    bigcn_tr_bu_fc = [list(range(bigcn_tr_bu_fc.shape[0])), bigcn_tr_bu_fc.sum(-1).tolist()]
    bigcn_ur_bu_fc = [list(range(bigcn_ur_bu_fc.shape[0])), bigcn_ur_bu_fc.sum(-1).tolist()]

    keys = list(ref_dict.keys())
    for key in keys:
        if key.find('conv') != -1 or key.find('fc') != -1:
            ref_dict.pop(key)

    ref_dict['nr_fc'] = bigcn_nr_fc
    ref_dict['fr_fc'] = bigcn_fr_fc
    ref_dict['tr_fc'] = bigcn_tr_fc
    ref_dict['ur_fc'] = bigcn_ur_fc

    ref_dict['nr_td_fc'] = bigcn_nr_td_fc
    ref_dict['fr_td_fc'] = bigcn_fr_td_fc
    ref_dict['tr_td_fc'] = bigcn_tr_td_fc
    ref_dict['ur_td_fc'] = bigcn_ur_td_fc

    ref_dict['nr_bu_fc'] = bigcn_nr_bu_fc
    ref_dict['fr_bu_fc'] = bigcn_fr_bu_fc
    ref_dict['tr_bu_fc'] = bigcn_tr_bu_fc
    ref_dict['ur_bu_fc'] = bigcn_ur_bu_fc

    ref_dict['lrp_class0_td_text'] = bigcn_lrp_class0_td_text
    ref_dict['lrp_class1_td_text'] = bigcn_lrp_class1_td_text
    ref_dict['lrp_class2_td_text'] = bigcn_lrp_class2_td_text
    ref_dict['lrp_class3_td_text'] = bigcn_lrp_class3_td_text
    ref_dict['lrp_allclass_td_text'] = bigcn_lrp_allclass_td_text

    ref_dict['lrp_class0_td_edge'] = bigcn_lrp_class0_td_edge
    ref_dict['lrp_class1_td_edge'] = bigcn_lrp_class1_td_edge
    ref_dict['lrp_class2_td_edge'] = bigcn_lrp_class2_td_edge
    ref_dict['lrp_class3_td_edge'] = bigcn_lrp_class3_td_edge
    ref_dict['lrp_allclass_td_edge'] = bigcn_lrp_allclass_td_edge

    ref_dict['lrp_class0_bu_text'] = bigcn_lrp_class0_bu_text
    ref_dict['lrp_class1_bu_text'] = bigcn_lrp_class1_bu_text
    ref_dict['lrp_class2_bu_text'] = bigcn_lrp_class2_bu_text
    ref_dict['lrp_class3_bu_text'] = bigcn_lrp_class3_bu_text
    ref_dict['lrp_allclass_bu_text'] = bigcn_lrp_allclass_bu_text

    ref_dict['lrp_class0_bu_edge'] = bigcn_lrp_class0_bu_edge
    ref_dict['lrp_class1_bu_edge'] = bigcn_lrp_class1_bu_edge
    ref_dict['lrp_class2_bu_edge'] = bigcn_lrp_class2_bu_edge
    ref_dict['lrp_class3_bu_edge'] = bigcn_lrp_class3_bu_edge
    ref_dict['lrp_allclass_bu_edge'] = bigcn_lrp_allclass_bu_edge


def process_cam_contributions(ref_dict):
    # TD
    bigcn_td_conv1_text = ref_dict['td_gcn_explanations']['conv1_text'][0]
    bigcn_td_conv2_text = ref_dict['td_gcn_explanations']['conv2_text'][0]
    bigcn_td_conv1_edge_weights = ref_dict['td_gcn_explanations']['conv1_edge_weights']
    bigcn_td_conv2_edge_weights = ref_dict['td_gcn_explanations']['conv2_edge_weights']

    bigcn_cam_class0_td_conv1 = ref_dict['cam_class0_td_conv1']
    bigcn_cam_class1_td_conv1 = ref_dict['cam_class1_td_conv1']
    bigcn_cam_class2_td_conv1 = ref_dict['cam_class2_td_conv1']
    bigcn_cam_class3_td_conv1 = ref_dict['cam_class3_td_conv1']

    bigcn_cam_class0_td_conv2 = ref_dict['cam_class0_td_conv2']
    bigcn_cam_class1_td_conv2 = ref_dict['cam_class1_td_conv2']
    bigcn_cam_class2_td_conv2 = ref_dict['cam_class2_td_conv2']
    bigcn_cam_class3_td_conv2 = ref_dict['cam_class3_td_conv2']

    logits = ref_dict['logits']
    logit_weights = np.exp(np.asarray(logits))[0]

    # Conv 1
    bigcn_cam_class0_td_conv1_text, bigcn_cam_class0_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_cam_class0_td_conv1)
    bigcn_cam_class1_td_conv1_text, bigcn_cam_class1_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_cam_class1_td_conv1)
    bigcn_cam_class2_td_conv1_text, bigcn_cam_class2_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_cam_class2_td_conv1)
    bigcn_cam_class3_td_conv1_text, bigcn_cam_class3_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_cam_class3_td_conv1)

    # Conv 2
    bigcn_cam_class0_td_conv2_text, bigcn_cam_class0_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_cam_class0_td_conv2)
    bigcn_cam_class1_td_conv2_text, bigcn_cam_class1_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_cam_class1_td_conv2)
    bigcn_cam_class2_td_conv2_text, bigcn_cam_class2_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_cam_class2_td_conv2)
    bigcn_cam_class3_td_conv2_text, bigcn_cam_class3_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_cam_class3_td_conv2)

    bigcn_cam_class0_td_text, t0 = get_text_ranked_list(bigcn_cam_class0_td_conv1_text,
                                                        bigcn_cam_class0_td_conv2_text)
    bigcn_cam_class1_td_text, t1 = get_text_ranked_list(bigcn_cam_class1_td_conv1_text,
                                                        bigcn_cam_class1_td_conv2_text)
    bigcn_cam_class2_td_text, t2 = get_text_ranked_list(bigcn_cam_class1_td_conv1_text,
                                                        bigcn_cam_class2_td_conv2_text)
    bigcn_cam_class3_td_text, t3 = get_text_ranked_list(bigcn_cam_class1_td_conv1_text,
                                                        bigcn_cam_class3_td_conv2_text)

    bigcn_cam_allclass_td_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                 np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    bigcn_cam_allclass_td_text /= bigcn_cam_allclass_td_text.sum()
    bigcn_cam_allclass_td_text = torch.topk(torch.as_tensor(bigcn_cam_allclass_td_text),
                                            k=bigcn_cam_allclass_td_text.shape[0])
    bigcn_cam_allclass_td_text = [bigcn_cam_allclass_td_text.indices.tolist(),
                                  bigcn_cam_allclass_td_text.values.tolist()]

    bigcn_cam_class0_td_edge, e0 = get_edge_ranked_list(bigcn_cam_class0_td_conv1_edge,
                                                        bigcn_cam_class0_td_conv2_edge)
    bigcn_cam_class1_td_edge, e1 = get_edge_ranked_list(bigcn_cam_class1_td_conv1_edge,
                                                        bigcn_cam_class1_td_conv2_edge)
    bigcn_cam_class2_td_edge, e2 = get_edge_ranked_list(bigcn_cam_class1_td_conv1_edge,
                                                        bigcn_cam_class2_td_conv2_edge)
    bigcn_cam_class3_td_edge, e3 = get_edge_ranked_list(bigcn_cam_class1_td_conv1_edge,
                                                        bigcn_cam_class3_td_conv2_edge)
    bigcn_cam_allclass_td_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                 np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    bigcn_cam_allclass_td_edge /= bigcn_cam_allclass_td_edge.sum()
    bigcn_cam_allclass_td_edge = torch.topk(torch.as_tensor(bigcn_cam_allclass_td_edge),
                                            k=bigcn_cam_allclass_td_edge.shape[0])
    bigcn_cam_allclass_td_edge = [bigcn_cam_allclass_td_edge.indices.tolist(),
                                  bigcn_cam_allclass_td_edge.values.tolist()]
    # BU
    bigcn_bu_conv1_text = ref_dict['bu_gcn_explanations']['conv1_text'][0]
    bigcn_bu_conv2_text = ref_dict['bu_gcn_explanations']['conv2_text'][0]
    bigcn_bu_conv1_edge_weights = ref_dict['bu_gcn_explanations']['conv1_edge_weights']
    bigcn_bu_conv2_edge_weights = ref_dict['bu_gcn_explanations']['conv2_edge_weights']

    bigcn_cam_class0_bu_conv1 = ref_dict['cam_class0_bu_conv1']
    bigcn_cam_class1_bu_conv1 = ref_dict['cam_class1_bu_conv1']
    bigcn_cam_class2_bu_conv1 = ref_dict['cam_class2_bu_conv1']
    bigcn_cam_class3_bu_conv1 = ref_dict['cam_class3_bu_conv1']

    bigcn_cam_class0_bu_conv2 = ref_dict['cam_class0_bu_conv2']
    bigcn_cam_class1_bu_conv2 = ref_dict['cam_class1_bu_conv2']
    bigcn_cam_class2_bu_conv2 = ref_dict['cam_class2_bu_conv2']
    bigcn_cam_class3_bu_conv2 = ref_dict['cam_class3_bu_conv2']

    logits = ref_dict['logits']
    logit_weights = np.exp(np.asarray(logits))[0]

    # Conv 1
    bigcn_cam_class0_bu_conv1_text, bigcn_cam_class0_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_cam_class0_bu_conv1)
    bigcn_cam_class1_bu_conv1_text, bigcn_cam_class1_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_cam_class1_bu_conv1)
    bigcn_cam_class2_bu_conv1_text, bigcn_cam_class2_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_cam_class2_bu_conv1)
    bigcn_cam_class3_bu_conv1_text, bigcn_cam_class3_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_cam_class3_bu_conv1)

    # Conv 2
    bigcn_cam_class0_bu_conv2_text, bigcn_cam_class0_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_cam_class0_bu_conv2)
    bigcn_cam_class1_bu_conv2_text, bigcn_cam_class1_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_cam_class1_bu_conv2)
    bigcn_cam_class2_bu_conv2_text, bigcn_cam_class2_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_cam_class2_bu_conv2)
    bigcn_cam_class3_bu_conv2_text, bigcn_cam_class3_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_cam_class3_bu_conv2)

    bigcn_cam_class0_bu_text, t0 = get_text_ranked_list(bigcn_cam_class0_bu_conv1_text,
                                                        bigcn_cam_class0_bu_conv2_text)
    bigcn_cam_class1_bu_text, t1 = get_text_ranked_list(bigcn_cam_class1_bu_conv1_text,
                                                        bigcn_cam_class1_bu_conv2_text)
    bigcn_cam_class2_bu_text, t2 = get_text_ranked_list(bigcn_cam_class1_bu_conv1_text,
                                                        bigcn_cam_class2_bu_conv2_text)
    bigcn_cam_class3_bu_text, t3 = get_text_ranked_list(bigcn_cam_class1_bu_conv1_text,
                                                        bigcn_cam_class3_bu_conv2_text)
    bigcn_cam_allclass_bu_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                 np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    bigcn_cam_allclass_bu_text /= bigcn_cam_allclass_bu_text.sum()
    bigcn_cam_allclass_bu_text = torch.topk(torch.as_tensor(bigcn_cam_allclass_bu_text),
                                            k=bigcn_cam_allclass_bu_text.shape[0])
    bigcn_cam_allclass_bu_text = [bigcn_cam_allclass_bu_text.indices.tolist(),
                                  bigcn_cam_allclass_bu_text.values.tolist()]

    bigcn_cam_class0_bu_edge, e0 = get_edge_ranked_list(bigcn_cam_class0_bu_conv1_edge,
                                                        bigcn_cam_class0_bu_conv2_edge)
    bigcn_cam_class1_bu_edge, e1 = get_edge_ranked_list(bigcn_cam_class1_bu_conv1_edge,
                                                        bigcn_cam_class1_bu_conv2_edge)
    bigcn_cam_class2_bu_edge, e2 = get_edge_ranked_list(bigcn_cam_class1_bu_conv1_edge,
                                                        bigcn_cam_class2_bu_conv2_edge)
    bigcn_cam_class3_bu_edge, e3 = get_edge_ranked_list(bigcn_cam_class1_bu_conv1_edge,
                                                        bigcn_cam_class3_bu_conv2_edge)
    bigcn_cam_allclass_bu_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                 np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    bigcn_cam_allclass_bu_edge /= bigcn_cam_allclass_bu_edge.sum()
    bigcn_cam_allclass_bu_edge = torch.topk(torch.as_tensor(bigcn_cam_allclass_bu_edge),
                                            k=bigcn_cam_allclass_bu_edge.shape[0])
    bigcn_cam_allclass_bu_edge = [bigcn_cam_allclass_bu_edge.indices.tolist(),
                                  bigcn_cam_allclass_bu_edge.values.tolist()]

    # Whole Model (Keep)
    # bigcn_cam_class0 = ref_dict['cam_class0_top_k']
    # bigcn_cam_class1 = ref_dict['cam_class1_top_k']
    # bigcn_cam_class2 = ref_dict['cam_class2_top_k']
    # bigcn_cam_class3 = ref_dict['cam_class3_top_k']
    # bigcn_cam_allclass = ref_dict['cam_allclass_top_k']

    # Extra
    bigcn_nr_fc = np.asarray(ref_dict['cam_class0_fc'])
    bigcn_fr_fc = np.asarray(ref_dict['cam_class1_fc'])
    bigcn_tr_fc = np.asarray(ref_dict['cam_class2_fc'])
    bigcn_ur_fc = np.asarray(ref_dict['cam_class3_fc'])

    bigcn_nr_fc = [bigcn_nr_fc[:bigcn_nr_fc.shape[-1] // 2].sum(),
                   bigcn_nr_fc[bigcn_nr_fc.shape[-1] // 2:].sum()]
    bigcn_fr_fc = [bigcn_fr_fc[:bigcn_fr_fc.shape[-1] // 2].sum(),
                   bigcn_fr_fc[bigcn_fr_fc.shape[-1] // 2:].sum()]
    bigcn_tr_fc = [bigcn_tr_fc[:bigcn_tr_fc.shape[-1] // 2].sum(),
                   bigcn_tr_fc[bigcn_tr_fc.shape[-1] // 2:].sum()]
    bigcn_ur_fc = [bigcn_ur_fc[:bigcn_ur_fc.shape[-1] // 2].sum(),
                   bigcn_ur_fc[bigcn_ur_fc.shape[-1] // 2:].sum()]

    bigcn_nr_td_fc = np.asarray(ref_dict['cam_class0_td_conv1_fc'])
    bigcn_fr_td_fc = np.asarray(ref_dict['cam_class1_td_conv1_fc'])
    bigcn_tr_td_fc = np.asarray(ref_dict['cam_class2_td_conv1_fc'])
    bigcn_ur_td_fc = np.asarray(ref_dict['cam_class3_td_conv1_fc'])
    bigcn_nr_td_fc = [list(range(bigcn_nr_td_fc.shape[0])), bigcn_nr_td_fc.sum(-1).tolist()]
    bigcn_fr_td_fc = [list(range(bigcn_fr_td_fc.shape[0])), bigcn_fr_td_fc.sum(-1).tolist()]
    bigcn_tr_td_fc = [list(range(bigcn_tr_td_fc.shape[0])), bigcn_tr_td_fc.sum(-1).tolist()]
    bigcn_ur_td_fc = [list(range(bigcn_ur_td_fc.shape[0])), bigcn_ur_td_fc.sum(-1).tolist()]

    bigcn_nr_bu_fc = np.asarray(ref_dict['cam_class0_bu_conv1_fc'])
    bigcn_fr_bu_fc = np.asarray(ref_dict['cam_class1_bu_conv1_fc'])
    bigcn_tr_bu_fc = np.asarray(ref_dict['cam_class2_bu_conv1_fc'])
    bigcn_ur_bu_fc = np.asarray(ref_dict['cam_class3_bu_conv1_fc'])
    bigcn_nr_bu_fc = [list(range(bigcn_nr_bu_fc.shape[0])), bigcn_nr_bu_fc.sum(-1).tolist()]
    bigcn_fr_bu_fc = [list(range(bigcn_fr_bu_fc.shape[0])), bigcn_fr_bu_fc.sum(-1).tolist()]
    bigcn_tr_bu_fc = [list(range(bigcn_tr_bu_fc.shape[0])), bigcn_tr_bu_fc.sum(-1).tolist()]
    bigcn_ur_bu_fc = [list(range(bigcn_ur_bu_fc.shape[0])), bigcn_ur_bu_fc.sum(-1).tolist()]

    keys = list(ref_dict.keys())
    for key in keys:
        if key.find('conv') != -1 or key.find('fc') != -1:
            ref_dict.pop(key)

    ref_dict['nr_fc'] = bigcn_nr_fc
    ref_dict['fr_fc'] = bigcn_fr_fc
    ref_dict['tr_fc'] = bigcn_tr_fc
    ref_dict['ur_fc'] = bigcn_ur_fc

    ref_dict['nr_td_fc'] = bigcn_nr_td_fc
    ref_dict['fr_td_fc'] = bigcn_fr_td_fc
    ref_dict['tr_td_fc'] = bigcn_tr_td_fc
    ref_dict['ur_td_fc'] = bigcn_ur_td_fc

    ref_dict['nr_bu_fc'] = bigcn_nr_bu_fc
    ref_dict['fr_bu_fc'] = bigcn_fr_bu_fc
    ref_dict['tr_bu_fc'] = bigcn_tr_bu_fc
    ref_dict['ur_bu_fc'] = bigcn_ur_bu_fc

    ref_dict['cam_class0_td_text'] = bigcn_cam_class0_td_text
    ref_dict['cam_class1_td_text'] = bigcn_cam_class1_td_text
    ref_dict['cam_class2_td_text'] = bigcn_cam_class2_td_text
    ref_dict['cam_class3_td_text'] = bigcn_cam_class3_td_text
    ref_dict['cam_allclass_td_text'] = bigcn_cam_allclass_td_text

    ref_dict['cam_class0_td_edge'] = bigcn_cam_class0_td_edge
    ref_dict['cam_class1_td_edge'] = bigcn_cam_class1_td_edge
    ref_dict['cam_class2_td_edge'] = bigcn_cam_class2_td_edge
    ref_dict['cam_class3_td_edge'] = bigcn_cam_class3_td_edge
    ref_dict['cam_allclass_td_edge'] = bigcn_cam_allclass_td_edge

    ref_dict['cam_class0_bu_text'] = bigcn_cam_class0_bu_text
    ref_dict['cam_class1_bu_text'] = bigcn_cam_class1_bu_text
    ref_dict['cam_class2_bu_text'] = bigcn_cam_class2_bu_text
    ref_dict['cam_class3_bu_text'] = bigcn_cam_class3_bu_text
    ref_dict['cam_allclass_bu_text'] = bigcn_cam_allclass_bu_text

    ref_dict['cam_class0_bu_edge'] = bigcn_cam_class0_bu_edge
    ref_dict['cam_class1_bu_edge'] = bigcn_cam_class1_bu_edge
    ref_dict['cam_class2_bu_edge'] = bigcn_cam_class2_bu_edge
    ref_dict['cam_class3_bu_edge'] = bigcn_cam_class3_bu_edge
    ref_dict['cam_allclass_bu_edge'] = bigcn_cam_allclass_bu_edge


def process_eb_contributions(ref_dict):
    # TD
    bigcn_td_conv1_text = ref_dict['td_gcn_explanations']['conv1_text'][0]
    bigcn_td_conv2_text = ref_dict['td_gcn_explanations']['conv2_text'][0]
    bigcn_td_conv1_edge_weights = ref_dict['td_gcn_explanations']['conv1_edge_weights']
    bigcn_td_conv2_edge_weights = ref_dict['td_gcn_explanations']['conv2_edge_weights']

    bigcn_eb_class0_td_conv1 = ref_dict['eb_class0_td_conv1']
    bigcn_eb_class1_td_conv1 = ref_dict['eb_class1_td_conv1']
    bigcn_eb_class2_td_conv1 = ref_dict['eb_class2_td_conv1']
    bigcn_eb_class3_td_conv1 = ref_dict['eb_class3_td_conv1']

    bigcn_eb_class0_td_conv2 = ref_dict['eb_class0_td_conv2']
    bigcn_eb_class1_td_conv2 = ref_dict['eb_class1_td_conv2']
    bigcn_eb_class2_td_conv2 = ref_dict['eb_class2_td_conv2']
    bigcn_eb_class3_td_conv2 = ref_dict['eb_class3_td_conv2']

    logits = ref_dict['logits']
    logit_weights = np.exp(np.asarray(logits))[0]

    # Conv 1
    bigcn_eb_class0_td_conv1_text, bigcn_eb_class0_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_eb_class0_td_conv1)
    bigcn_eb_class1_td_conv1_text, bigcn_eb_class1_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_eb_class1_td_conv1)
    bigcn_eb_class2_td_conv1_text, bigcn_eb_class2_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_eb_class2_td_conv1)
    bigcn_eb_class3_td_conv1_text, bigcn_eb_class3_td_conv1_edge, _ = compute_gcl_relevance(
        bigcn_td_conv1_text,
        *bigcn_td_conv1_edge_weights, bigcn_eb_class3_td_conv1)

    # Conv 2
    bigcn_eb_class0_td_conv2_text, bigcn_eb_class0_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_eb_class0_td_conv2)
    bigcn_eb_class1_td_conv2_text, bigcn_eb_class1_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_eb_class1_td_conv2)
    bigcn_eb_class2_td_conv2_text, bigcn_eb_class2_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_eb_class2_td_conv2)
    bigcn_eb_class3_td_conv2_text, bigcn_eb_class3_td_conv2_edge, _ = compute_gcl_relevance(
        bigcn_td_conv2_text,
        *bigcn_td_conv2_edge_weights, bigcn_eb_class3_td_conv2)

    bigcn_eb_class0_td_text, t0 = get_text_ranked_list(bigcn_eb_class0_td_conv1_text,
                                                        bigcn_eb_class0_td_conv2_text)
    bigcn_eb_class1_td_text, t1 = get_text_ranked_list(bigcn_eb_class1_td_conv1_text,
                                                        bigcn_eb_class1_td_conv2_text)
    bigcn_eb_class2_td_text, t2 = get_text_ranked_list(bigcn_eb_class1_td_conv1_text,
                                                        bigcn_eb_class2_td_conv2_text)
    bigcn_eb_class3_td_text, t3 = get_text_ranked_list(bigcn_eb_class1_td_conv1_text,
                                                        bigcn_eb_class3_td_conv2_text)

    bigcn_eb_allclass_td_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                 np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    bigcn_eb_allclass_td_text /= bigcn_eb_allclass_td_text.sum()
    bigcn_eb_allclass_td_text = torch.topk(torch.as_tensor(bigcn_eb_allclass_td_text),
                                            k=bigcn_eb_allclass_td_text.shape[0])
    bigcn_eb_allclass_td_text = [bigcn_eb_allclass_td_text.indices.tolist(),
                                  bigcn_eb_allclass_td_text.values.tolist()]

    bigcn_eb_class0_td_edge, e0 = get_edge_ranked_list(bigcn_eb_class0_td_conv1_edge,
                                                        bigcn_eb_class0_td_conv2_edge)
    bigcn_eb_class1_td_edge, e1 = get_edge_ranked_list(bigcn_eb_class1_td_conv1_edge,
                                                        bigcn_eb_class1_td_conv2_edge)
    bigcn_eb_class2_td_edge, e2 = get_edge_ranked_list(bigcn_eb_class1_td_conv1_edge,
                                                        bigcn_eb_class2_td_conv2_edge)
    bigcn_eb_class3_td_edge, e3 = get_edge_ranked_list(bigcn_eb_class1_td_conv1_edge,
                                                        bigcn_eb_class3_td_conv2_edge)
    bigcn_eb_allclass_td_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                 np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    bigcn_eb_allclass_td_edge /= bigcn_eb_allclass_td_edge.sum()
    bigcn_eb_allclass_td_edge = torch.topk(torch.as_tensor(bigcn_eb_allclass_td_edge),
                                            k=bigcn_eb_allclass_td_edge.shape[0])
    bigcn_eb_allclass_td_edge = [bigcn_eb_allclass_td_edge.indices.tolist(),
                                  bigcn_eb_allclass_td_edge.values.tolist()]
    # BU
    bigcn_bu_conv1_text = ref_dict['bu_gcn_explanations']['conv1_text'][0]
    bigcn_bu_conv2_text = ref_dict['bu_gcn_explanations']['conv2_text'][0]
    bigcn_bu_conv1_edge_weights = ref_dict['bu_gcn_explanations']['conv1_edge_weights']
    bigcn_bu_conv2_edge_weights = ref_dict['bu_gcn_explanations']['conv2_edge_weights']

    bigcn_eb_class0_bu_conv1 = ref_dict['eb_class0_bu_conv1']
    bigcn_eb_class1_bu_conv1 = ref_dict['eb_class1_bu_conv1']
    bigcn_eb_class2_bu_conv1 = ref_dict['eb_class2_bu_conv1']
    bigcn_eb_class3_bu_conv1 = ref_dict['eb_class3_bu_conv1']

    bigcn_eb_class0_bu_conv2 = ref_dict['eb_class0_bu_conv2']
    bigcn_eb_class1_bu_conv2 = ref_dict['eb_class1_bu_conv2']
    bigcn_eb_class2_bu_conv2 = ref_dict['eb_class2_bu_conv2']
    bigcn_eb_class3_bu_conv2 = ref_dict['eb_class3_bu_conv2']

    logits = ref_dict['logits']
    logit_weights = np.exp(np.asarray(logits))[0]

    # Conv 1
    bigcn_eb_class0_bu_conv1_text, bigcn_eb_class0_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_eb_class0_bu_conv1)
    bigcn_eb_class1_bu_conv1_text, bigcn_eb_class1_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_eb_class1_bu_conv1)
    bigcn_eb_class2_bu_conv1_text, bigcn_eb_class2_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_eb_class2_bu_conv1)
    bigcn_eb_class3_bu_conv1_text, bigcn_eb_class3_bu_conv1_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv1_text,
        *bigcn_bu_conv1_edge_weights, bigcn_eb_class3_bu_conv1)

    # Conv 2
    bigcn_eb_class0_bu_conv2_text, bigcn_eb_class0_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_eb_class0_bu_conv2)
    bigcn_eb_class1_bu_conv2_text, bigcn_eb_class1_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_eb_class1_bu_conv2)
    bigcn_eb_class2_bu_conv2_text, bigcn_eb_class2_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_eb_class2_bu_conv2)
    bigcn_eb_class3_bu_conv2_text, bigcn_eb_class3_bu_conv2_edge, _ = compute_gcl_relevance(
        bigcn_bu_conv2_text,
        *bigcn_bu_conv2_edge_weights, bigcn_eb_class3_bu_conv2)

    bigcn_eb_class0_bu_text, t0 = get_text_ranked_list(bigcn_eb_class0_bu_conv1_text,
                                                        bigcn_eb_class0_bu_conv2_text)
    bigcn_eb_class1_bu_text, t1 = get_text_ranked_list(bigcn_eb_class1_bu_conv1_text,
                                                        bigcn_eb_class1_bu_conv2_text)
    bigcn_eb_class2_bu_text, t2 = get_text_ranked_list(bigcn_eb_class1_bu_conv1_text,
                                                        bigcn_eb_class2_bu_conv2_text)
    bigcn_eb_class3_bu_text, t3 = get_text_ranked_list(bigcn_eb_class1_bu_conv1_text,
                                                        bigcn_eb_class3_bu_conv2_text)
    bigcn_eb_allclass_bu_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                 np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    bigcn_eb_allclass_bu_text /= bigcn_eb_allclass_bu_text.sum()
    bigcn_eb_allclass_bu_text = torch.topk(torch.as_tensor(bigcn_eb_allclass_bu_text),
                                            k=bigcn_eb_allclass_bu_text.shape[0])
    bigcn_eb_allclass_bu_text = [bigcn_eb_allclass_bu_text.indices.tolist(),
                                  bigcn_eb_allclass_bu_text.values.tolist()]

    bigcn_eb_class0_bu_edge, e0 = get_edge_ranked_list(bigcn_eb_class0_bu_conv1_edge,
                                                        bigcn_eb_class0_bu_conv2_edge)
    bigcn_eb_class1_bu_edge, e1 = get_edge_ranked_list(bigcn_eb_class1_bu_conv1_edge,
                                                        bigcn_eb_class1_bu_conv2_edge)
    bigcn_eb_class2_bu_edge, e2 = get_edge_ranked_list(bigcn_eb_class1_bu_conv1_edge,
                                                        bigcn_eb_class2_bu_conv2_edge)
    bigcn_eb_class3_bu_edge, e3 = get_edge_ranked_list(bigcn_eb_class1_bu_conv1_edge,
                                                        bigcn_eb_class3_bu_conv2_edge)
    bigcn_eb_allclass_bu_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                 np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    bigcn_eb_allclass_bu_edge /= bigcn_eb_allclass_bu_edge.sum()
    bigcn_eb_allclass_bu_edge = torch.topk(torch.as_tensor(bigcn_eb_allclass_bu_edge),
                                            k=bigcn_eb_allclass_bu_edge.shape[0])
    bigcn_eb_allclass_bu_edge = [bigcn_eb_allclass_bu_edge.indices.tolist(),
                                  bigcn_eb_allclass_bu_edge.values.tolist()]

    # Whole Model (Keep)
    # bigcn_eb_class0 = ref_dict['eb_class0_top_k']
    # bigcn_eb_class1 = ref_dict['eb_class1_top_k']
    # bigcn_eb_class2 = ref_dict['eb_class2_top_k']
    # bigcn_eb_class3 = ref_dict['eb_class3_top_k']
    # bigcn_eb_allclass = ref_dict['eb_allclass_top_k']

    # Extra
    bigcn_nr_fc = np.asarray(ref_dict['eb_class0_fc'])
    bigcn_fr_fc = np.asarray(ref_dict['eb_class1_fc'])
    bigcn_tr_fc = np.asarray(ref_dict['eb_class2_fc'])
    bigcn_ur_fc = np.asarray(ref_dict['eb_class3_fc'])

    bigcn_nr_fc = [bigcn_nr_fc[:bigcn_nr_fc.shape[-1] // 2].sum(),
                   bigcn_nr_fc[bigcn_nr_fc.shape[-1] // 2:].sum()]
    bigcn_fr_fc = [bigcn_fr_fc[:bigcn_fr_fc.shape[-1] // 2].sum(),
                   bigcn_fr_fc[bigcn_fr_fc.shape[-1] // 2:].sum()]
    bigcn_tr_fc = [bigcn_tr_fc[:bigcn_tr_fc.shape[-1] // 2].sum(),
                   bigcn_tr_fc[bigcn_tr_fc.shape[-1] // 2:].sum()]
    bigcn_ur_fc = [bigcn_ur_fc[:bigcn_ur_fc.shape[-1] // 2].sum(),
                   bigcn_ur_fc[bigcn_ur_fc.shape[-1] // 2:].sum()]

    bigcn_nr_td_fc = np.asarray(ref_dict['eb_class0_td_conv1_fc'])
    bigcn_fr_td_fc = np.asarray(ref_dict['eb_class1_td_conv1_fc'])
    bigcn_tr_td_fc = np.asarray(ref_dict['eb_class2_td_conv1_fc'])
    bigcn_ur_td_fc = np.asarray(ref_dict['eb_class3_td_conv1_fc'])
    bigcn_nr_td_fc = [list(range(bigcn_nr_td_fc.shape[0])), bigcn_nr_td_fc.sum(-1).tolist()]
    bigcn_fr_td_fc = [list(range(bigcn_fr_td_fc.shape[0])), bigcn_fr_td_fc.sum(-1).tolist()]
    bigcn_tr_td_fc = [list(range(bigcn_tr_td_fc.shape[0])), bigcn_tr_td_fc.sum(-1).tolist()]
    bigcn_ur_td_fc = [list(range(bigcn_ur_td_fc.shape[0])), bigcn_ur_td_fc.sum(-1).tolist()]

    bigcn_nr_bu_fc = np.asarray(ref_dict['eb_class0_bu_conv1_fc'])
    bigcn_fr_bu_fc = np.asarray(ref_dict['eb_class1_bu_conv1_fc'])
    bigcn_tr_bu_fc = np.asarray(ref_dict['eb_class2_bu_conv1_fc'])
    bigcn_ur_bu_fc = np.asarray(ref_dict['eb_class3_bu_conv1_fc'])
    bigcn_nr_bu_fc = [list(range(bigcn_nr_bu_fc.shape[0])), bigcn_nr_bu_fc.sum(-1).tolist()]
    bigcn_fr_bu_fc = [list(range(bigcn_fr_bu_fc.shape[0])), bigcn_fr_bu_fc.sum(-1).tolist()]
    bigcn_tr_bu_fc = [list(range(bigcn_tr_bu_fc.shape[0])), bigcn_tr_bu_fc.sum(-1).tolist()]
    bigcn_ur_bu_fc = [list(range(bigcn_ur_bu_fc.shape[0])), bigcn_ur_bu_fc.sum(-1).tolist()]

    keys = list(ref_dict.keys())
    for key in keys:
        if key.find('conv') != -1 or key.find('fc') != -1:
            ref_dict.pop(key)

    ref_dict['nr_fc'] = bigcn_nr_fc
    ref_dict['fr_fc'] = bigcn_fr_fc
    ref_dict['tr_fc'] = bigcn_tr_fc
    ref_dict['ur_fc'] = bigcn_ur_fc

    ref_dict['nr_td_fc'] = bigcn_nr_td_fc
    ref_dict['fr_td_fc'] = bigcn_fr_td_fc
    ref_dict['tr_td_fc'] = bigcn_tr_td_fc
    ref_dict['ur_td_fc'] = bigcn_ur_td_fc

    ref_dict['nr_bu_fc'] = bigcn_nr_bu_fc
    ref_dict['fr_bu_fc'] = bigcn_fr_bu_fc
    ref_dict['tr_bu_fc'] = bigcn_tr_bu_fc
    ref_dict['ur_bu_fc'] = bigcn_ur_bu_fc

    ref_dict['eb_class0_td_text'] = bigcn_eb_class0_td_text
    ref_dict['eb_class1_td_text'] = bigcn_eb_class1_td_text
    ref_dict['eb_class2_td_text'] = bigcn_eb_class2_td_text
    ref_dict['eb_class3_td_text'] = bigcn_eb_class3_td_text
    ref_dict['eb_allclass_td_text'] = bigcn_eb_allclass_td_text

    ref_dict['eb_class0_td_edge'] = bigcn_eb_class0_td_edge
    ref_dict['eb_class1_td_edge'] = bigcn_eb_class1_td_edge
    ref_dict['eb_class2_td_edge'] = bigcn_eb_class2_td_edge
    ref_dict['eb_class3_td_edge'] = bigcn_eb_class3_td_edge
    ref_dict['eb_allclass_td_edge'] = bigcn_eb_allclass_td_edge

    ref_dict['eb_class0_bu_text'] = bigcn_eb_class0_bu_text
    ref_dict['eb_class1_bu_text'] = bigcn_eb_class1_bu_text
    ref_dict['eb_class2_bu_text'] = bigcn_eb_class2_bu_text
    ref_dict['eb_class3_bu_text'] = bigcn_eb_class3_bu_text
    ref_dict['eb_allclass_bu_text'] = bigcn_eb_allclass_bu_text

    ref_dict['eb_class0_bu_edge'] = bigcn_eb_class0_bu_edge
    ref_dict['eb_class1_bu_edge'] = bigcn_eb_class1_bu_edge
    ref_dict['eb_class2_bu_edge'] = bigcn_eb_class2_bu_edge
    ref_dict['eb_class3_bu_edge'] = bigcn_eb_class3_bu_edge
    ref_dict['eb_allclass_bu_edge'] = bigcn_eb_allclass_bu_edge


def extract_intermediates_bigcn_twitter1516(conv1, conv2, x, edge_index, data_sample, device):
    gcn_explanations = {}
    edge_weight = None

    edge_index, edge_weight = gcn_conv.gcn_norm(  # yapf: disable
        edge_index, edge_weight, x.size(-2), False, True)
    conv1_lin_output = conv1.lin(x)
    # conv1_attribution = torch.zeros((x.shape[0])).to(device)
    # for edge_num, (src, dst) in enumerate(zip(*edge_index)):
    #     src, dst = src.item(), dst.item()
    #     conv1_attribution[src] += (abs(conv1_lin_output[src]) * edge_weight[edge_num]).sum()
    # print(conv1_attribution)
    # conv1_attribution_top_k = torch.topk(conv1_attribution, k=conv1_attribution.shape[0])
    # gcn_explanations['conv1_attribution_top_k'] = [conv1_attribution_top_k.indices.tolist(),
    #                                                conv1_attribution_top_k.values.tolist()]
    gcn_explanations['conv1_edge_weights'] = [edge_index.tolist(), edge_weight.tolist()]
    gcn_explanations['conv1_text'] = [conv1_lin_output.tolist()]

    conv1_output = conv1(x, edge_index)
    # conv1_output_sum = torch.sum(conv1_output, dim=-1)
    # print('conv1_output_sum', conv1_output_sum)
    # conv1_output_sum_topk = torch.topk(conv1_output_sum,
    #                                    k=conv1_output_sum.shape[0])
    # print('conv1_output_sum_topk', conv1_output_sum_topk)
    # gcn_explanations['conv1_output_sum_topk'] = \
    #     [conv1_output_sum_topk.indices.tolist(),
    #      conv1_output_sum_topk.values.tolist()]

    rootindex = data_sample.rootindex
    root_extend = torch.zeros(len(data_sample.batch), x.size(1)).to(device)
    batch_size = max(data_sample.batch) + 1
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend[index] = x[rootindex[num_batch]]
    conv1_output_cat_x = torch.cat((conv1_output, root_extend), 1)
    # conv1_output_cat_x_sum = torch.sum(conv1_output_cat_x, dim=-1)
    # print('conv1_output_cat_x_sum', conv1_output_cat_x_sum)
    # conv1_output_cat_x_sum_topk = torch.topk(conv1_output_cat_x_sum,
    #                                          k=conv1_output_cat_x_sum.shape[0])
    # print('conv1_output_cat_x_sum_topk', conv1_output_cat_x_sum_topk)
    # gcn_explanations['conv1_output_cat_x_sum_topk'] = \
    #     [conv1_output_cat_x_sum_topk.indices.tolist(),
    #      conv1_output_cat_x_sum_topk.values.tolist()]

    conv1_output_cat_x_relu = F.relu(conv1_output_cat_x)
    conv2_input = F.dropout(conv1_output_cat_x_relu, training=False)
    # conv2_input_sum = torch.sum(conv2_input, dim=-1)
    # print('conv2_input_sum', conv2_input_sum)
    # conv2_input_sum_topk = torch.topk(conv2_input_sum,
    #                                   k=conv2_input_sum.shape[0])
    # print('conv2_input_sum_topk', conv2_input_sum_topk)
    # gcn_explanations['conv2_input_sum_topk'] = \
    #     [conv2_input_sum_topk.indices.tolist(),
    #      conv2_input_sum_topk.values.tolist()]

    conv1_output_cat_x_text_only = torch.cat((conv1_lin_output, root_extend), 1)
    conv1_output_cat_x_relu_text_only = F.relu(conv1_output_cat_x_text_only)
    conv2_input_text_only = F.dropout(conv1_output_cat_x_relu_text_only, training=False)
    # conv2_input_sum = torch.sum(conv2_input_text_only, dim=-1)

    edge_index, edge_weight = gcn_conv.gcn_norm(  # yapf: disable
        edge_index, edge_weight, x.size(-2), False, True)
    # conv2_lin_output = conv2.lin(conv2_input)
    # conv2_attribution = torch.zeros((x.shape[0])).to(device)
    # for edge_num, (src, dst) in enumerate(zip(*edge_index)):
    #     src, dst = src.item(), dst.item()
    #     conv2_attribution[src] += (abs(conv2_lin_output[src]) * edge_weight[edge_num]).sum()
    # print(conv2_attribution)
    # conv2_attribution_top_k = torch.topk(conv2_attribution, k=conv2_attribution.shape[0])
    # gcn_explanations['conv2_attribution_top_k'] = [conv2_attribution_top_k.indices.tolist(),
    #                                                conv2_attribution_top_k.values.tolist()]
    conv2_lin_output_text_only = conv2.lin(conv2_input_text_only)
    gcn_explanations['conv2_edge_weights'] = [edge_index.tolist(), edge_weight.tolist()]
    gcn_explanations['conv2_text'] = [conv2_lin_output_text_only.tolist()]

    conv2_output = conv2(conv2_input, edge_index)
    # conv2_output_sum = torch.sum(conv2_output, dim=-1)
    # print('conv2_output_sum', conv2_output_sum)
    # conv2_output_sum_topk = torch.topk(conv2_output_sum,
    #                                           k=conv2_output_sum.shape[0])
    # print('conv2_output_sum_topk', conv2_output_sum_topk)
    # gcn_explanations['conv2_output_sum_topk'] = \
    #     [conv2_output_sum_topk.indices.tolist(),
    #      conv2_output_sum_topk.values.tolist()]

    conv2_output_relu = F.relu(conv2_output)

    root_extend2 = torch.zeros(len(data_sample.batch), conv1_output.size(1)).to(device)
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend2[index] = conv1_output[rootindex[num_batch]]
    scatter_mean_input = torch.cat((conv2_output_relu, root_extend2), 1)
    # scatter_mean_input_sum = torch.sum(scatter_mean_input, dim=-1)
    # print('scatter_mean_input_sum', scatter_mean_input_sum)
    # scatter_mean_input_sum_topk = torch.topk(scatter_mean_input_sum,
    #                                                 k=scatter_mean_input_sum.shape[0])
    # gcn_explanations['scatter_mean_input_sum_topk'] = \
    #     [scatter_mean_input_sum_topk.indices.tolist(),
    #      scatter_mean_input_sum_topk.values.tolist()]

    gcn_output = scatter_mean(scatter_mean_input, data_sample.batch, dim=0)
    # gcn_output_sum = torch.sum(gcn_output, dim=-1)
    # print('gcn_output', gcn_output)
    gcn_explanations['gcn_output'] = gcn_output.tolist()

    return gcn_explanations


def extract_intermediates_ebgcn_twitter1516(gcn, x, edge_index, data_sample, device):
    gcn_explanations = {}
    edge_weight = None

    edge_index, edge_weight = gcn_conv.gcn_norm(  # yapf: disable
        edge_index, edge_weight, x.size(-2), False, True)
    conv1_lin_output = gcn.conv1.lin(x)
    # conv1_attribution = torch.zeros((x.shape[0])).to(device)
    # for edge_num, (src, dst) in enumerate(zip(*edge_index)):
    #     src, dst = src.item(), dst.item()
    #     conv1_attribution[src] += (abs(conv1_lin_output[src]) * edge_weight[edge_num]).sum()
    # print(conv1_attribution)
    # conv1_attribution_top_k = torch.topk(conv1_attribution, k=conv1_attribution.shape[0])
    # gcn_explanations['conv1_attribution_top_k'] = [conv1_attribution_top_k.indices.tolist(),
    #                                                conv1_attribution_top_k.values.tolist()]
    gcn_explanations['conv1_edge_weights'] = [edge_index.tolist(), edge_weight.tolist()]
    gcn_explanations['conv1_text'] = [conv1_lin_output.tolist()]

    conv1_output = gcn.conv1(x, edge_index)
    # conv1_output_sum = torch.sum(conv1_output, dim=-1)
    # conv1_output_sum_topk = torch.topk(conv1_output_sum,
    #                                    k=conv1_output_sum.shape[0])
    # gcn_explanations['conv1_output_sum_topk'] = \
    #     [conv1_output_sum_topk.indices.tolist(),
    #      conv1_output_sum_topk.values.tolist()]

    edge_loss, edge_pred = gcn.edge_infer(conv1_output, edge_index)
    # if gcn.args.edge_infer_td:
    #     edge_loss, edge_pred = gcn.edge_infer(x, edge_index)
    # else:
    #     edge_loss, edge_pred = None, None
    # edge_loss, edge_pred = None, None

    rootindex = data_sample.rootindex
    root_extend = torch.zeros(len(data_sample.batch), x.size(1)).to(device)
    batch_size = max(data_sample.batch) + 1
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend[index] = x[rootindex[num_batch]]
    conv1_output_cat_x = torch.cat((conv1_output, root_extend), 1)

    # conv1_output_cat_x_sum = torch.sum(conv1_output_cat_x, dim=-1)
    # conv1_output_cat_x_sum_topk = torch.topk(conv1_output_cat_x_sum,
    #                                          k=conv1_output_cat_x_sum.shape[0])
    # gcn_explanations['conv1_output_cat_x_sum_topk'] = \
    #     [conv1_output_cat_x_sum_topk.indices.tolist(),
    #      conv1_output_cat_x_sum_topk.values.tolist()]

    conv1_output_cat_x_bn1 = gcn.bn1(conv1_output_cat_x)
    # conv1_output_cat_x_bn1_sum = torch.sum(conv1_output_cat_x_bn1, dim=-1)
    # conv1_output_cat_x_bn1_sum_topk = torch.topk(conv1_output_cat_x_bn1_sum,
    #                                          k=conv1_output_cat_x_bn1_sum.shape[0])
    # gcn_explanations['conv1_output_cat_x_bn1_sum_topk'] = \
    #     [conv1_output_cat_x_bn1_sum_topk.indices.tolist(),
    #      conv1_output_cat_x_bn1_sum_topk.values.tolist()]

    conv1_output_cat_x_bn1_relu = F.relu(conv1_output_cat_x_bn1)

    conv1_output_cat_x_text_only = torch.cat((conv1_lin_output, root_extend), 1)
    conv1_output_cat_x_relu_text_only = F.relu(conv1_output_cat_x_text_only)
    conv2_input_text_only = F.dropout(conv1_output_cat_x_relu_text_only, training=False)
    conv1_output_cat_x_bn1_text_only = gcn.bn1(conv1_output_cat_x_text_only)
    conv1_output_cat_x_bn1_relu_text_only = F.relu(conv1_output_cat_x_bn1_text_only)
    conv2_input_text_only = conv1_output_cat_x_bn1_relu_text_only

    conv2_input = conv1_output_cat_x_bn1_relu
    # conv2_input_sum = torch.sum(conv2_input, dim=-1)
    # conv2_input_sum_topk = torch.topk(conv2_input_sum,
    #                                   k=conv2_input_sum.shape[0])
    # gcn_explanations['conv2_input_sum_topk'] = \
    #     [conv2_input_sum_topk.indices.tolist(),
    #      conv2_input_sum_topk.values.tolist()]

    # conv2_lin_output = gcn.conv2.lin(conv2_input)
    # conv2_attribution = torch.zeros((x.shape[0])).to(device)
    # for edge_num, (src, dst) in enumerate(zip(*edge_index)):
    #     src, dst = src.item(), dst.item()
    #     conv2_attribution[src] += (abs(conv2_lin_output[src]) * edge_pred[edge_num]).sum()
    # print(conv2_attribution)
    # conv2_attribution_top_k = torch.topk(conv2_attribution, k=conv2_attribution.shape[0])
    # gcn_explanations['conv2_attribution_top_k'] = [conv2_attribution_top_k.indices.tolist(),
    #                                                conv2_attribution_top_k.values.tolist()]
    conv2_lin_output_text_only = gcn.conv2.lin(conv2_input_text_only)
    gcn_explanations['conv2_edge_weights'] = [edge_index.tolist(), edge_pred.tolist()]
    gcn_explanations['conv2_text'] = [conv2_lin_output_text_only.tolist()]

    conv2_output = gcn.conv2(conv2_input, edge_index, edge_weight=edge_pred)
    # conv2_output_sum = torch.sum(conv2_output, dim=-1)
    # conv2_output_sum_topk = torch.topk(conv2_output_sum,
    #                                    k=conv2_output_sum.shape[0])
    # gcn_explanations['conv2_output_sum_topk'] = \
    #     [conv2_output_sum_topk.indices.tolist(),
    #      conv2_output_sum_topk.values.tolist()]

    conv2_output_relu = F.relu(conv2_output)

    root_extend2 = torch.zeros(len(data_sample.batch), conv1_output.size(1)).to(device)
    for num_batch in range(batch_size):
        index = (torch.eq(data_sample.batch, num_batch))
        root_extend2[index] = conv1_output[rootindex[num_batch]]
    conv2_output_relu_cat_conv1_output = torch.cat((conv2_output_relu, root_extend2), 1)
    scatter_mean_input = conv2_output_relu_cat_conv1_output
    # scatter_mean_input_sum = torch.sum(scatter_mean_input, dim=-1)
    # scatter_mean_input_sum_topk = torch.topk(scatter_mean_input_sum,
    #                                          k=scatter_mean_input_sum.shape[0])
    # gcn_explanations['scatter_mean_input_sum_topk'] = \
    #     [scatter_mean_input_sum_topk.indices.tolist(),
    #      scatter_mean_input_sum_topk.values.tolist()]

    gcn_output = scatter_mean(scatter_mean_input, data_sample.batch, dim=0)
    gcn_explanations['gcn_output'] = gcn_output.tolist()

    return gcn_explanations


def enumerate_children(module: nn.Module):
    for child in module.children():
        enumerate_children(child)
        print(child)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname', type=str, default="Twitter16", metavar='dataname',
                        help='dataset name')
    parser.add_argument('--modelname', type=str, default="BiGCN", metavar='modeltype',
                        help='model type, option: BiGCN/EBGCN')
    parser.add_argument('--input_features', type=int, default=5000, metavar='inputF',
                        help='dimension of input features (TF-IDF)')
    parser.add_argument('--hidden_features', type=int, default=64, metavar='graph_hidden',
                        help='dimension of graph hidden state')
    parser.add_argument('--output_features', type=int, default=64, metavar='output_features',
                        help='dimension of output features')
    parser.add_argument('--num_class', type=int, default=4, metavar='numclass',
                        help='number of classes')
    parser.add_argument('--num_workers', type=int, default=0, metavar='num_workers',
                        help='number of workers for training')

    # Parameters for training the model
    parser.add_argument('--seed', type=int, default=2020, help='random state seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='does not use GPU')
    parser.add_argument('--num_cuda', type=int, default=0,
                        help='index of GPU 0/1')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_scale_bu', type=int, default=5, metavar='LRSB',
                        help='learning rate scale for bottom-up direction')
    parser.add_argument('--lr_scale_td', type=int, default=1, metavar='LRST',
                        help='learning rate scale for top-down direction')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2',
                        help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--patience', type=int, default=10, metavar='patience',
                        help='patience for early stop')
    parser.add_argument('--batchsize', type=int, default=128, metavar='BS',
                        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='E',
                        help='number of max epochs')
    parser.add_argument('--iterations', type=int, default=50, metavar='F',
                        help='number of iterations for 5-fold cross-validation')

    # Parameters for the proposed model
    parser.add_argument('--TDdroprate', type=float, default=0, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0, metavar='BUdroprate',
                        help='drop rate for edges in the bottom-up dispersion graph')
    parser.add_argument('--edge_infer_td', action='store_true', default=True,  # default=False,
                        help='edge inference in the top-down graph')
    parser.add_argument('--edge_infer_bu', action='store_true', default=True,  # default=True,
                        help='edge inference in the bottom-up graph')
    parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the top-down propagation graph')
    parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the bottom-up dispersion graph')
    parser.add_argument('--edge_num', type=int, default=2, metavar='edgenum',
                        help='latent relation types T in the edge inference')

    args = parser.parse_args()

    # some admin stuff
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TDdroprate = 0  # 0.2
    BUdroprate = 0  # 0.2
    # datasetname = sys.argv[1]  # "Twitter15"、"Twitter16", 'PHEME'
    datasetname = 'Twitter'  # 'Twitter', 'PHEME'
    # iterations = int(sys.argv[2])
    iterations = 1
    # if datasetname == 'Twitter':
    batchsize = 1
    args.datasetname = datasetname
    # args.input_features = 256 * 768
    args.batchsize = batchsize
    args.iterations = iterations
    args.device = device
    args.training = False

    split_type = '5fold'  # ['5fold']
    model_types = ['BiGCN', 'EBGCN']
    # pooling_types = ['max', 'mean']
    randomise_types = [0.25]  # [1.0, 0.75, 0.5, 0.25]
    drop_edge_types = [1]  # [1.0, 0.75, 0.5, 0.25]
    set_initial_weight_types = [0.5, 2.0]  # [0.5, 1.0, 2.0]
    config_type = 'original'  # ['original', 'nodes', 'edges', 'weights']
    version = 2
    model = model_types[0]
    # pooling = pooling_types[0]
    # drop_edges = randomise_types[2]  # Randomise Nodes
    treeDic = None  # Not required for PHEME
    min_graph_size = 20  # Exclude graphs with less that this number of nodes
    exp_method = 'cam'  # ['lrp', 'cam', 'eb']
    LRP_PARAMS['mode'] = exp_method

    # save_dir_path = 'data/explain/Twitter'
    SAVE_DIR_PATH = os.path.join(EXPLAIN_DIR, datasetname, exp_method)
    if not os.path.exists(SAVE_DIR_PATH):
        os.makedirs(SAVE_DIR_PATH)

    # init network
    if model == 'BiGCN':
        net = BiGCN(5000, 64, 64, device).to(device)
        checkpoint_paths = ['best_5fold-BiGCNv2_Twitter_f0_i0_e00018_l0.08147.pt',
                            'best_5fold-BiGCNv2_Twitter_f1_i0_e00013_l0.15382.pt',
                            'best_5fold-BiGCNv2_Twitter_f2_i0_e00012_l0.19403.pt',
                            'best_5fold-BiGCNv2_Twitter_f3_i0_e00017_l0.07888.pt',
                            'best_5fold-BiGCNv2_Twitter_f4_i0_e00013_l0.15599.pt']
    elif model == 'EBGCN':
        args.input_features = 5000
        checkpoint_paths = ['best_5fold-EBGCNv2_Twitter_f0_i0_e00004_l0.32038.pt',
                            'best_5fold-EBGCNv2_Twitter_f1_i0_e00004_l0.33221.pt',
                            'best_5fold-EBGCNv2_Twitter_f2_i0_e00005_l0.19204.pt',
                            'best_5fold-EBGCNv2_Twitter_f3_i0_e00005_l0.19187.pt',
                            'best_5fold-EBGCNv2_Twitter_f4_i0_e00004_l0.32603.pt']
        net = EBGCN(args).to(device)

    def reinit_net():
        if model == 'BiGCN':
            net = BiGCN(5000, 64, 64, device).to(device)
        elif model == 'EBGCN':
            args.input_features = 5000
            net = EBGCN(args).to(device)
        return net

    # Get generated list
    # for fold_num, fold in enumerate(load5foldData(datasetname)):
    #     if fold_num % 2 != 0:  # Training fold, skip this
    #         continue
    #     else:
    #         fold_num = fold_num // 2

    outputs = []
    if 2 <= version <= 3:
        model0 = f'{model}v{version}'
    else:
        model0 = model
    model0 = f'{split_type}-{model0}'
    # temp_file_path = 'temp_file.txt'
    if config_type == 'original':
        print(f'\nGenerating:\t'
              f'Model: {model0}\t'
              f'Original')
        if split_type == '5fold':
            for fold_num, fold in enumerate(load5foldData(datasetname)):
                if fold_num % 2 != 0:  # Training fold, skip this
                    continue
                else:
                    fold_num = fold_num // 2
                    # if fold_num != 1:  # MemoryError
                    #     continue
                net = reinit_net()
                try:
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, datasetname, checkpoint_paths[fold_num])
                    checkpoint = torch.load(checkpoint_path)
                    net.load_state_dict(checkpoint['model_state_dict'])
                    print(f'Checkpoint loaded from {checkpoint_path}')
                except:
                    print('No checkpoint to load')
                net.eval()

                model_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                model_copy.eval()
                if exp_method == 'eb':  # Copy contrastive model
                    net_copy = copy.deepcopy(net)
                    contrastive_model = lrp_utils.get_lrpwrappermodule(net_copy, LRP_PARAMS, is_contrastive=True)
                    contrastive_model.eval()
                fold_output = {}

                event_name = f'fold{fold_num}'
                if not os.path.exists(os.path.join(SAVE_DIR_PATH, event_name)):
                    os.makedirs(os.path.join(SAVE_DIR_PATH, event_name))
                    print(f'Save directory for {event_name} created at: '
                          f'{os.path.join(SAVE_DIR_PATH, event_name)}\n')
                else:
                    print(f'Save directory for {event_name} already exists at: '
                          f'{os.path.join(SAVE_DIR_PATH, event_name)}\n')
                fold_test = fold
                fold_train = []
                treeDic = loadTree(datasetname)
                traindata_list, testdata_list = loadBiData(datasetname,
                                                           treeDic,
                                                           fold_train,
                                                           fold_test,
                                                           TDdroprate,
                                                           BUdroprate)
                test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=5,
                                         pin_memory=False)
                evaluation_log_path = os.path.join(EXPLAIN_DIR,
                                                   f'{datasetname}_{event_name}_{model0}_original_eval_'
                                                   f'{exp_method}.txt')
                eval_log_string = ''
                conf_mat = np.zeros((4, 4))
                num_skipped = 0
                total_num = 0
                for sample_num, (data_sample, root_tweetid) in enumerate(tqdm(test_loader)):
                    total_num += 1

                    explain_output = {}
                    # print(type(data_sample['edge_index']), isinstance(data_sample['edge_index'], torch_sparse.SparseTensor))
                    # data_sample.retains_grad = True
                    data_sample = data_sample.to(device)
                    if 2 <= version <= 3:
                        try:
                            data_sample.x = data_sample.cls
                        except:
                            pass
                    edge_index = data_sample.edge_index
                    BU_edge_index = data_sample.BU_edge_index
                    # tweetids = data_sample.tweetids
                    # explain_output['tweetids'] = tweetids.tolist()

                    # node_num_to_tweetid = {}
                    # for node_num, tweetid in enumerate(tweetids):
                    #     tweetid: torch.Tensor
                    #     node_num_to_tweetid[int(node_num)] = tweetid.item()
                    #
                    # explain_output['node_num_to_tweetid'] = node_num_to_tweetid

                    explain_output['rootindex'] = data_sample.rootindex.item()

                    # x_sum = torch.sum(data_sample.x, dim=-1)
                    # print('x_sum', x_sum)
                    # x_sum_top_k = torch.topk(x_sum,
                    #                          k=x_sum.shape[0])
                    # explain_output['x_sum_top_k'] = [x_sum_top_k.indices.tolist(),
                    #                                  x_sum_top_k.values.tolist()]

                    # TODO: Need to finish the extract method
                    if model == 'BiGCN':
                        if 2 <= version <= 3:
                            try:
                                data_sample.x = data_sample.cls
                            except:
                                pass
                        x = data_sample.x
                        # TD
                        td_gcn_conv1 = net.TDrumorGCN.conv1
                        td_gcn_conv2 = net.TDrumorGCN.conv2
                        td_gcn_explanations = extract_intermediates_bigcn_twitter1516(td_gcn_conv1, td_gcn_conv2, x,
                                                                                      edge_index, data_sample, device)
                        explain_output['td_gcn_explanations'] = td_gcn_explanations
                        # BU
                        bu_gcn_conv1 = net.BUrumorGCN.conv1
                        bu_gcn_conv2 = net.BUrumorGCN.conv2
                        bu_gcn_explanations = extract_intermediates_bigcn_twitter1516(bu_gcn_conv1, bu_gcn_conv2, x,
                                                                                      BU_edge_index, data_sample, device)
                        explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                        # bigcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                        # bigcn_copy.eval()
                        relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                        for target in range(4):
                            data_copy = copy.deepcopy(data_sample)
                            data_copy.x.requires_grad = True
                            model_copy.zero_grad()
                            if exp_method == 'eb':  # Extra data copy for contrastive model
                                data_copy2 = copy.deepcopy(data_sample)
                                data_copy2.x.requires_grad = True
                                contrastive_model.zero_grad()
                            with torch.enable_grad():
                                temp_output = model_copy(data_copy)
                                temp_output[0, target].backward()
                                relevance = data_copy.x.grad.sum(-1)
                                if exp_method == 'eb':  # Backprop through contrastive model
                                    temp_output2 = contrastive_model(data_copy2)
                                    temp_output2[0, target].backward()
                                    relevance2 = data_copy2.x.grad.sum(-1)
                            for key, val in model_copy.saved_rels.items():
                                if exp_method == 'lrp' or exp_method == 'cam':
                                    explain_output[f'{exp_method}_class{target}_{key}'] = val.tolist()
                                elif exp_method == 'eb':  # Obtain contrastive result
                                    contrastive_val = contrastive_model.saved_rels.get(key, None)
                                    if contrastive_val is None:
                                        print('Error in contrastive values')
                                        raise Exception
                                    new_val = val - contrastive_val
                                    explain_output[f'eb_class{target}_{key}'] = new_val.tolist()
                            if exp_method == 'eb':  # Obtain contrastive result
                                contrastive_result = (relevance * torch.exp(temp_output[0, target])) - \
                                                     (relevance2 * torch.exp(temp_output2[0, target]))
                                relevance_all_classes[:, target] = contrastive_result.cpu()
                                relevance_top_k = torch.topk(contrastive_result, k=contrastive_result.shape[0])
                            else:
                                relevance_all_classes[:, target] = (relevance * torch.exp(temp_output[0, target])).cpu()
                                relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                            # print(root_tweetid, target, relevance_top_k)
                            explain_output[f'{exp_method}_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                   relevance_top_k.values.tolist()]
                        else:
                            relevance_all_classes = torch.sum(relevance_all_classes, -1)
                            relevance_all_classes /= relevance_all_classes.sum()
                            relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                            explain_output[f'{exp_method}_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                              relevance_top_k.values.tolist()]
                        out_labels = net(data_sample)
                    elif model == 'EBGCN':
                        if 2 <= version <= 3:
                            try:
                                data_sample.x = data_sample.cls
                            except:
                                pass
                        x = data_sample.x
                        # TD
                        td_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.TDrumorGCN, x, edge_index,
                                                                                      data_sample, device)
                        explain_output['td_gcn_explanations'] = td_gcn_explanations
                        # BU
                        bu_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.BUrumorGCN, x, BU_edge_index,
                                                                                      data_sample, device)
                        explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                        # ebgcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                        # ebgcn_copy.eval()
                        relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                        for target in range(4):
                            data_copy = copy.deepcopy(data_sample)
                            data_copy.x.requires_grad = True
                            model_copy.zero_grad()
                            if exp_method == 'eb':  # Extra data copy for contrastive model
                                data_copy2 = copy.deepcopy(data_sample)
                                data_copy2.x.requires_grad = True
                                contrastive_model.zero_grad()
                            with torch.enable_grad():
                                temp_output, _, _ = model_copy(data_copy)
                                temp_output[0, target].backward()
                                relevance = data_copy.x.grad.sum(-1)
                                if exp_method == 'eb':  # Backprop through contrastive model
                                    temp_output2, _, _ = contrastive_model(data_copy2)
                                    temp_output2[0, target].backward()
                                    relevance2 = data_copy2.x.grad.sum(-1)
                            for key, val in model_copy.saved_rels.items():
                                if exp_method == 'lrp' or exp_method == 'cam':
                                    explain_output[f'{exp_method}_class{target}_{key}'] = val.tolist()
                                elif exp_method == 'eb':  # Obtain contrastive result
                                    contrastive_val = contrastive_model.saved_rels.get(key, None)
                                    if contrastive_val is None:
                                        print('Error in contrastive values')
                                        raise Exception
                                    new_val = val - contrastive_val
                                    explain_output[f'eb_class{target}_{key}'] = new_val.tolist()
                            if exp_method == 'eb':  # Obtain contrastive result
                                contrastive_result = (relevance * torch.exp(temp_output[0, target])) - \
                                                     (relevance2 * torch.exp(temp_output2[0, target]))
                                relevance_all_classes[:, target] = contrastive_result.cpu()
                                relevance_top_k = torch.topk(contrastive_result, k=contrastive_result.shape[0])
                            else:
                                relevance_all_classes[:, target] = (
                                        relevance * torch.exp(temp_output[0, target])).cpu()
                                relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                            # print(root_tweetid, target, relevance_top_k)
                            explain_output[f'{exp_method}_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                   relevance_top_k.values.tolist()]
                        else:
                            relevance_all_classes = torch.sum(relevance_all_classes, -1)
                            relevance_all_classes /= relevance_all_classes.sum()
                            relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                            explain_output[f'{exp_method}_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                              relevance_top_k.values.tolist()]
                        out_labels, _, _ = net(data_sample)
                    _, pred = out_labels.max(dim=-1)
                    correct = pred.eq(data_sample.y).sum().item()
                    # print(pred.item(), data_sample.y.item(), correct)
                    explain_output['logits'] = out_labels.tolist()
                    explain_output['prediction'] = pred.item()
                    explain_output['ground_truth'] = data_sample.y.item()
                    explain_output['correct_prediction'] = correct
                    # Process LRP maps to reduce size
                    if exp_method == 'lrp':
                        process_lrp_contributions(explain_output)
                    elif exp_method == 'cam':
                        process_cam_contributions(explain_output)
                    elif exp_method == 'eb':
                        process_eb_contributions(explain_output)
                    if data_sample.x.shape[0] >= min_graph_size:
                        eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\n'
                    else:
                        eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\t' \
                                           f'Skipped\n'
                    conf_mat[data_sample.y.item(), pred.item()] += 1
                    if data_sample.x.shape[0] < min_graph_size:
                        num_skipped += 1
                        continue
                    fold_output[int(root_tweetid[0][0])] = explain_output
                    save = True
                    if save:
                        save_path = os.path.join(SAVE_DIR_PATH, event_name,
                                                 f'{root_tweetid[0][0]}_{model0}_original_explain.json')
                        # print(f'Saving to {save_path}')
                        try:
                            with open(save_path, 'w') as f:
                                json.dump(explain_output, f, indent=1)
                            # print(f'\rSaved to {save_path}')
                        except:
                            print(f'\rFailed to save to {save_path}')
                    # print('test done')
                    # raise Exception
                outputs.append(fold_output)
                total_evaluated = conf_mat.sum()
                total_correct = conf_mat.diagonal().sum()
                acc = total_correct / total_evaluated
                eval_log_string += f'Skipped: {num_skipped / total_num * 100:.2f}% [{num_skipped}]/[{total_num}]\n'
                eval_log_string += f'Acc: {acc * 100:.5f}% [{total_correct}]/[{total_evaluated}]\n'
                for i in range(4):
                    precision = conf_mat[i, i] / conf_mat[:, i].sum()
                    recall = conf_mat[i, i] / conf_mat[i, :].sum()
                    f1 = 2 * precision * recall / (precision + recall)
                    eval_log_string += f'Class {i}:\t' \
                                       f'Precision: {precision}\t' \
                                       f'Recall: {recall}\t' \
                                       f'F1: {f1}\n'
                eval_log_string += f' {"":20} | {"":20} | {"Predicted":20}\n' \
                                   f' {"":20} | {"":20} | {"Class 0":20} | {"Class 1":20} | {"Class 2":20} | {"Class 3":20}\n'
                for i in range(4):
                    if i != 0:
                        eval_log_string += f' {"":20} | {f"Class {i}":20} |'
                    else:
                        eval_log_string += f' {"Actual":20} | {f"Class {i}":20} |'
                    eval_log_string += f' {conf_mat[i, 0]:20} | {conf_mat[i, 1]:20} |' \
                                       f' {conf_mat[i, 2]:20} | {conf_mat[i, 3]:20}\n'
                save = True
                if save:
                    with open(evaluation_log_path, 'w') as f:
                        f.write(eval_log_string)
    elif config_type == 'nodes':
        for randomise in randomise_types:
            print(f'\nGenerating:\t'
                  f'Model: {model0}\t'
                  f'Randomise Nodes: {randomise}')
            if split_type == '5fold':
                for fold_num, fold in enumerate(load5foldData(datasetname)):
                    if fold_num % 2 != 0:  # Training fold, skip this
                        continue
                    else:
                        fold_num = fold_num // 2
                        # if fold_num != 0:  # MemoryError
                        #     continue
                    c0_train, c0_test, c1_train, c1_test, c2_train, c2_test, c3_train, c3_test = fold
                    fold_train = np.concatenate((c0_train, c1_train, c2_train, c3_train)).tolist()
                    fold_test = np.concatenate((c0_test, c1_test, c2_test, c3_test)).tolist()
                    try:
                        checkpoint_path = os.path.join(CHECKPOINT_DIR, datasetname, checkpoint_paths[fold_num])
                        checkpoint = torch.load(checkpoint_path)
                        net.load_state_dict(checkpoint['model_state_dict'])
                        print(f'Checkpoint loaded from {checkpoint_path}')
                    except:
                        print('No checkpoint to load')
                    net.eval()

                    model_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                    model_copy.eval()
                    fold_output = {}

                    event_name = f'fold{fold_num}'
                    if not os.path.exists(os.path.join(EXPLAIN_DIR, datasetname, event_name)):
                        os.makedirs(os.path.join(EXPLAIN_DIR, datasetname, event_name))
                        print(f'Save directory for {event_name} created at: '
                              f'{os.path.join(EXPLAIN_DIR, datasetname, event_name)}\n')
                    else:
                        print(f'Save directory for {event_name} already exists\n')
                    traindata_list, testdata_list = loadBiData(datasetname,
                                                               treeDic,
                                                               fold_train,
                                                               fold_test,
                                                               TDdroprate,
                                                               BUdroprate)
                    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=5)
                    evaluation_log_path = os.path.join(EXPLAIN_DIR,
                                                       f'{datasetname}_{event_name}_{model0}_n{randomise}_eval_'
                                                       f'{exp_method}.txt')
                    eval_log_string = ''
                    conf_mat = np.zeros((4, 4))
                    num_skipped = 0
                    total_num = 0
                    for sample_num, (data_sample, root_tweetid) in enumerate(tqdm(test_loader)):
                        total_num += 1

                        explain_output = {}
                        # print(type(data_sample['edge_index']), isinstance(data_sample['edge_index'], torch_sparse.SparseTensor))
                        # data_sample.retains_grad = True
                        data_sample = data_sample.to(device)

                        lrp = False
                        # TODO: Finish this
                        if lrp:
                            # data_sample.x = data_sample.x.reshape(data_sample.x.shape[0], -1, 768).sum(1)
                            # print(data_sample.edge_index.requires_grad, data_sample.BU_edge_index.requires_grad)
                            # gcn_copy = lrp_utils.get_lrpwrappermodule(net.TDrumorGCN.conv1, LRP_PARAMS)
                            # with torch.enable_grad():
                            #     temp_output = gcn_copy(temp_x, data_sample.edge_index)
                            #     temp_output[0, 0].backward()
                            # print(net.TDrumorGCN.bn1.running_var,
                            #       net.TDrumorGCN.bn1.weight, net.TDrumorGCN.bn1.bias,
                            #       net.TDrumorGCN.bn1.running_mean)
                            # print(net.TDrumorGCN.bn1.eps, net.TDrumorGCN.bn1.momentum)
                            # tdrumourgcn_copy = lrp_utils.get_lrpwrappermodule(net.TDrumorGCN, LRP_PARAMS)
                            # with torch.enable_grad():
                            #     temp_output, _ = tdrumourgcn_copy(data_sample)
                            #     temp_output[0, 0].backward()
                            ebgcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            # bigcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            data_sample.x.requires_grad = True
                            with torch.enable_grad():
                                temp_output, _, _ = ebgcn_copy(data_sample)
                                # temp_output = bigcn_copy(data_sample)
                                target = temp_output.argmax().item()
                                temp_output[0, target].backward()
                                # print(torch.eq(data_sample.x, data_sample.x.grad))
                                print(data_sample.x.grad.sum(-1))
                                if sample_num == 4:
                                    raise Exception
                            continue

                        if 2 <= version <= 3:
                            data_sample.x = data_sample.cls
                        x = data_sample.x
                        if randomise == 1.0:
                            indices = random.sample(range(x.shape[0]), x.shape[0])
                            x = swap_elements(x, range(x.shape[0]), indices)
                            explain_output['swapped_nodes'] = [list(range(x.shape[0])), indices]
                        elif randomise != 0.0:
                            sample_len = int(x.shape[0] * randomise // 1)
                            if sample_len % 2 != 0 and sample_len > 1 and x.shape[0] > 1:
                                sample_len -= 1
                            indices = random.sample(range(x.shape[0]), sample_len)
                            # print(indices[:int(sample_len/2)], indices[int(sample_len/2):])
                            x = swap_elements(x, indices[:int(sample_len/2)], indices[int(sample_len/2):])
                            explain_output['swapped_nodes'] = [indices[:int(sample_len/2)], indices[int(sample_len/2):]]
                        data_sample.x = x
                        edge_index = data_sample.edge_index
                        BU_edge_index = data_sample.BU_edge_index
                        tweetids = data_sample.tweetids
                        explain_output['tweetids'] = tweetids.tolist()

                        node_num_to_tweetid = {}
                        for node_num, tweetid in enumerate(tweetids):
                            tweetid: torch.Tensor
                            node_num_to_tweetid[int(node_num)] = tweetid.item()

                        explain_output['node_num_to_tweetid'] = node_num_to_tweetid

                        explain_output['rootindex'] = data_sample.rootindex.item()

                        x_sum = torch.sum(data_sample.x, dim=-1)
                        # print('x_sum', x_sum)
                        x_sum_top_k = torch.topk(x_sum,
                                                 k=x_sum.shape[0])
                        # if x.shape[0] != len(node_num_to_tweetid) \
                        #         or x.shape[0] != x_sum.shape[0] \
                        #         or x.shape[0] != x_sum_top_k.indices.shape[0] \
                        #         or x_sum.shape[0] != x_sum_top_k.indices.shape[0]:
                        #     print('Error')
                            # print(root_tweetid, len(tweetids), x.shape, x_sum.shape, x_sum_top_k.indices.shape)
                            # raise Exception
                        # print(root_tweetid, len(node_num_to_tweetid), x.shape, x_sum.shape, x_sum_top_k.indices.shape)
                        # print('x_sum_top_k', x_sum_top_k)
                        explain_output['x_sum_top_k'] = [x_sum_top_k.indices.tolist(),
                                                         x_sum_top_k.values.tolist()]

                        # TODO: Need to finish the extract method
                        if model == 'BiGCN':
                            # if version == 2:  # Version 2
                            #     new_x = x
                            #     new_x = new_x.reshape(new_x.shape[0], -1, 768)
                            #     new_x = new_x[:, 0]
                            #     x = new_x
                            #     data_sample.x = x
                            if 2 <= version <= 3:
                                data_sample.x = data_sample.cls
                            # TD
                            td_gcn_conv1 = net.TDrumorGCN.conv1
                            td_gcn_conv2 = net.TDrumorGCN.conv2
                            td_gcn_explanations = extract_intermediates_bigcn_twitter1516(td_gcn_conv1, td_gcn_conv2, x,
                                                                                          edge_index, data_sample, device)
                            explain_output['td_gcn_explanations'] = td_gcn_explanations
                            # BU
                            bu_gcn_conv1 = net.BUrumorGCN.conv1
                            bu_gcn_conv2 = net.BUrumorGCN.conv2
                            bu_gcn_explanations = extract_intermediates_bigcn_twitter1516(bu_gcn_conv1, bu_gcn_conv2, x,
                                                                                          BU_edge_index, data_sample, device)
                            explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                            # bigcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            # bigcn_copy.eval()
                            relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                            for target in range(4):
                                data_copy = copy.deepcopy(data_sample)
                                data_copy.x.requires_grad = True
                                model_copy.zero_grad()
                                with torch.enable_grad():
                                    temp_output = model_copy(data_copy)
                                    temp_output[0, target].backward()
                                    relevance = data_copy.x.grad.sum(-1)
                                relevance_all_classes[:, target] = (relevance * torch.exp(temp_output[0, target])).cpu()
                                relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                                # print(root_tweetid, target, relevance_top_k)
                                explain_output[f'lrp_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                              relevance_top_k.values.tolist()]
                            else:
                                relevance_all_classes = torch.sum(relevance_all_classes, -1)
                                relevance_all_classes /= relevance_all_classes.sum()
                                relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                                explain_output[f'lrp_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                         relevance_top_k.values.tolist()]
                            out_labels = net(data_sample)
                        elif model == 'EBGCN':
                            # if version == 2:  # Version 2
                            #     new_x = x
                            #     new_x = new_x.reshape(new_x.shape[0], -1, 768)
                            #     new_x = new_x[:, 0]
                            #     x = new_x
                            #     data_sample.x = x
                            if 2 <= version <= 3:
                                data_sample.x = data_sample.cls
                            # TD
                            td_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.TDrumorGCN, x, edge_index, data_sample, device)
                            explain_output['td_gcn_explanations'] = td_gcn_explanations
                            #BU
                            bu_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.BUrumorGCN, x, BU_edge_index, data_sample, device)
                            explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                            # ebgcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            # ebgcn_copy.eval()
                            relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                            for target in range(4):
                                data_copy = copy.deepcopy(data_sample)
                                data_copy.x.requires_grad = True
                                model_copy.zero_grad()
                                with torch.enable_grad():
                                    temp_output, _, _ = model_copy(data_copy)
                                    temp_output[0, target].backward()
                                    relevance = data_copy.x.grad.sum(-1)
                                relevance_all_classes[:, target] = (relevance * torch.exp(temp_output[0, target])).cpu()
                                relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                                # print(root_tweetid, target, relevance_top_k)
                                explain_output[f'lrp_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                              relevance_top_k.values.tolist()]
                            else:
                                relevance_all_classes = torch.sum(relevance_all_classes, -1)
                                relevance_all_classes /= relevance_all_classes.sum()
                                relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                                explain_output[f'lrp_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                         relevance_top_k.values.tolist()]
                                # if sample_num == 4:
                                #     raise Exception
                                # continue
                            out_labels, _, _ = net(data_sample)

                        _, pred = out_labels.max(dim=-1)
                        correct = pred.eq(data_sample.y).sum().item()
                        # print(pred.item(), data_sample.y.item(), correct)
                        explain_output['logits'] = out_labels.tolist()
                        explain_output['prediction'] = pred.item()
                        explain_output['ground_truth'] = data_sample.y.item()
                        explain_output['correct_prediction'] = correct
                        if data_sample.x.shape[0] < min_graph_size:
                            eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\n'
                        else:
                            eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\t' \
                                               f'Skipped\n'
                        conf_mat[data_sample.y.item(), pred.item()] += 1
                        if data_sample.x.shape[0] < min_graph_size:
                            num_skipped += 1
                            continue
                        fold_output[int(root_tweetid[0][0])] = explain_output
                        save = True
                        if save:
                            with open(os.path.join(EXPLAIN_DIR, datasetname, event_name,
                                                   f'{root_tweetid[0][0]}_{model0}_n{randomise}_explain.json'), 'w') as f:
                                json.dump(explain_output, f, indent=1)
                        # raise Exception
                    # break
                    outputs.append(fold_output)
                    total_evaluated = conf_mat.sum()
                    total_correct = conf_mat.diagonal().sum()
                    acc = total_correct/total_evaluated
                    eval_log_string += f'Skipped: {num_skipped / total_num * 100:.2f}% [{num_skipped}]/[{total_num}]\n'
                    eval_log_string += f'Acc: {acc*100:.5f}% [{total_correct}]/[{total_evaluated}]\n'
                    for i in range(4):
                        precision = conf_mat[i, i] / conf_mat[:, i].sum()
                        recall = conf_mat[i, i] / conf_mat[i, :].sum()
                        f1 = 2 * precision * recall / (precision + recall)
                        eval_log_string += f'Class {i}:\t' \
                                           f'Precision: {precision}\t' \
                                           f'Recall: {recall}\t' \
                                           f'F1: {f1}\n'
                    eval_log_string += f' {"":20} | {"":20} | {"Predicted":20}\n' \
                                       f' {"":20} | {"":20} | {"Class 0":20} | {"Class 1":20} | {"Class 2":20} | {"Class 3":20}\n'
                    for i in range(4):
                        if i != 0:
                            eval_log_string += f' {"":20} | {f"Class {i}":20} |'
                        else:
                            eval_log_string += f' {"Actual":20} | {f"Class {i}":20} |'
                        eval_log_string += f' {conf_mat[i, 0]:20} | {conf_mat[i, 1]:20} |' \
                                           f' {conf_mat[i, 2]:20} | {conf_mat[i, 3]:20}\n'
                    save = True
                    if save:
                        with open(evaluation_log_path, 'w') as f:
                            f.write(eval_log_string)
    elif config_type == 'edges':
        for drop_edges in drop_edge_types:
            print(f'\nGenerating:\t'
                  f'Model: {model0}\t'
                  f'Drop Edges: {drop_edges}')
            if split_type == '5fold':
                for fold_num, fold in enumerate(load5foldData(datasetname)):
                    if fold_num % 2 != 0:  # Training fold, skip this
                        continue
                    else:
                        fold_num = fold_num // 2
                        # if fold_num != 0:  # MemoryError
                        #     continue
                    net = reinit_net()
                    try:
                        checkpoint_path = os.path.join(CHECKPOINT_DIR, datasetname, checkpoint_paths[fold_num])
                        checkpoint = torch.load(checkpoint_path)
                        net.load_state_dict(checkpoint['model_state_dict'])
                        print(f'Checkpoint loaded from {checkpoint_path}')
                    except:
                        print('No checkpoint to load')
                    net.eval()

                    model_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                    model_copy.eval()
                    if exp_method == 'eb':  # Copy contrastive model
                        net_copy = copy.deepcopy(net)
                        contrastive_model = lrp_utils.get_lrpwrappermodule(net_copy, LRP_PARAMS, is_contrastive=True)
                        contrastive_model.eval()
                    fold_output = {}

                    event_name = f'fold{fold_num}'
                    if not os.path.exists(os.path.join(SAVE_DIR_PATH, event_name)):
                        os.makedirs(os.path.join(SAVE_DIR_PATH, event_name))
                        print(f'Save directory for {event_name} created at: '
                              f'{os.path.join(SAVE_DIR_PATH, event_name)}\n')
                    else:
                        print(f'Save directory for {event_name} already exists at: '
                              f'{os.path.join(SAVE_DIR_PATH, event_name)}\n')
                    fold_test = fold
                    fold_train = []
                    treeDic = loadTree(datasetname)
                    traindata_list, testdata_list = loadBiData(datasetname,
                                                               treeDic,
                                                               fold_train,
                                                               fold_test,
                                                               TDdroprate,
                                                               BUdroprate)
                    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=5)
                    evaluation_log_path = os.path.join(EXPLAIN_DIR,
                                                       f'{datasetname}_{event_name}_{model0}_d{drop_edges}_eval_'
                                                       f'{exp_method}.txt')
                    eval_log_string = ''
                    conf_mat = np.zeros((4, 4))
                    num_skipped = 0
                    total_num = 0
                    for sample_num, (data_sample, root_tweetid) in enumerate(tqdm(test_loader)):
                        total_num += 1
                        # print(root_tweetid, data_sample.x.shape)
                        explain_output = {}
                        # print(type(data_sample['edge_index']), isinstance(data_sample['edge_index'], torch_sparse.SparseTensor))
                        # data_sample.retains_grad = True
                        data_sample = data_sample.to(device)

                        if 2 <= version <= 3:
                            try:
                                data_sample.x = data_sample.cls
                            except:
                                pass
                        edge_index = data_sample.edge_index
                        BU_edge_index = data_sample.BU_edge_index
                        if drop_edges == 1.0:
                            explain_output['dropped_edges'] = old_edge_list = list(map(lambda u, v: f'{u}-{v}',
                                                                                       edge_index[0].tolist(),
                                                                                       edge_index[1].tolist()))
                            new_edge_index: torch.Tensor = torch.zeros((2, 0),
                                                                       dtype=edge_index.dtype)
                            new_BU_edge_index: torch.Tensor = torch.zeros_like(new_edge_index,
                                                                               dtype=edge_index.dtype)
                            # print(new_edge_index, new_BU_edge_index, old_edge_list)
                            # raise Exception
                        elif drop_edges != 0.0:
                            sample_len = int(edge_index.shape[1] * (1 - drop_edges) // 1)
                            # if sample_len % 2 != 0 and sample_len > 1 and x.shape[0] > 1:
                            #     sample_len -= 1
                            # print(sample_len, edge_index.shape[1])
                            indices = torch.LongTensor(sorted(random.sample(range(edge_index.shape[1]), sample_len)))
                            new_edge_index: torch.Tensor = torch.zeros((2, len(indices)),
                                                                       dtype=edge_index.dtype)
                            new_BU_edge_index: torch.Tensor = torch.zeros_like(new_edge_index,
                                                                               dtype=edge_index.dtype)
                            # print(edge_index[0], edge_index[0].shape, indices, indices.shape)
                            new_edge_index[0] = torch.index_select(edge_index[0].cpu(), 0, indices)
                            new_edge_index[1] = torch.index_select(edge_index[1].cpu(), 0, indices)
                            new_BU_edge_index[0] = torch.index_select(BU_edge_index[0].cpu(), 0, indices)
                            new_BU_edge_index[1] = torch.index_select(BU_edge_index[1].cpu(), 0, indices)
                            old_edge_list = list(map(lambda u, v: f'{u}-{v}',
                                                     edge_index[0].tolist(),
                                                     edge_index[1].tolist()))
                            new_edge_list = list(map(lambda u, v: f'{u}-{v}',
                                                     new_edge_index[0].tolist(),
                                                     new_edge_index[1].tolist()))
                            explain_output['dropped_edges'] = list(sorted(set(old_edge_list) ^ set(new_edge_list)))
                            # print(new_edge_index, new_BU_edge_index, explain_output['dropped_edges'])
                        data_sample.edge_index = new_edge_index.to(device)
                        data_sample.BU_edge_index = new_BU_edge_index.to(device)
                        edge_index = data_sample.edge_index
                        BU_edge_index = data_sample.BU_edge_index
                        # tweetids = data_sample.tweetids
                        # explain_output['tweetids'] = tweetids.tolist()

                        # node_num_to_tweetid = {}
                        # for node_num, tweetid in enumerate(tweetids):
                        #     tweetid: torch.Tensor
                        #     node_num_to_tweetid[int(node_num)] = tweetid.item()
                        #
                        # explain_output['node_num_to_tweetid'] = node_num_to_tweetid

                        explain_output['rootindex'] = data_sample.rootindex.item()

                        # x_sum = torch.sum(data_sample.x, dim=-1)
                        # print('x_sum', x_sum)
                        # x_sum_top_k = torch.topk(x_sum,
                        #                          k=x_sum.shape[0])
                        # if x.shape[0] != len(node_num_to_tweetid) \
                        #         or x.shape[0] != x_sum.shape[0] \
                        #         or x.shape[0] != x_sum_top_k.indices.shape[0] \
                        #         or x_sum.shape[0] != x_sum_top_k.indices.shape[0]:
                        #     print('Error')
                        # print(root_tweetid, len(tweetids), x.shape, x_sum.shape, x_sum_top_k.indices.shape)
                        # raise Exception
                        # print(root_tweetid, len(node_num_to_tweetid), x.shape, x_sum.shape, x_sum_top_k.indices.shape)
                        # print('x_sum_top_k', x_sum_top_k)
                        # explain_output['x_sum_top_k'] = [x_sum_top_k.indices.tolist(),
                        #                                  x_sum_top_k.values.tolist()]

                        # TODO: Need to finish the extract method
                        if model == 'BiGCN':
                            # if version == 2:  # Version 2
                            #     new_x = x
                            #     new_x = new_x.reshape(new_x.shape[0], -1, 768)
                            #     new_x = new_x[:, 0]
                            #     x = new_x
                            #     data_sample.x = x
                            if 2 <= version <= 3:
                                try:
                                    data_sample.x = data_sample.cls
                                except:
                                    pass
                            x = data_sample.x
                            # TD
                            td_gcn_conv1 = net.TDrumorGCN.conv1
                            td_gcn_conv2 = net.TDrumorGCN.conv2
                            td_gcn_explanations = extract_intermediates_bigcn_twitter1516(td_gcn_conv1, td_gcn_conv2, x,
                                                                                          edge_index, data_sample, device)
                            explain_output['td_gcn_explanations'] = td_gcn_explanations
                            # BU
                            bu_gcn_conv1 = net.BUrumorGCN.conv1
                            bu_gcn_conv2 = net.BUrumorGCN.conv2
                            bu_gcn_explanations = extract_intermediates_bigcn_twitter1516(bu_gcn_conv1, bu_gcn_conv2, x,
                                                                                          BU_edge_index, data_sample, device)
                            explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                            # bigcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            # bigcn_copy.eval()
                            relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                            for target in range(4):
                                data_copy = copy.deepcopy(data_sample)
                                data_copy.x.requires_grad = True
                                model_copy.zero_grad()
                                if exp_method == 'eb':  # Extra data copy for contrastive model
                                    data_copy2 = copy.deepcopy(data_sample)
                                    data_copy2.x.requires_grad = True
                                    contrastive_model.zero_grad()
                                with torch.enable_grad():
                                    temp_output = model_copy(data_copy)
                                    temp_output[0, target].backward()
                                    relevance = data_copy.x.grad.sum(-1)
                                    if exp_method == 'eb':  # Backprop through contrastive model
                                        temp_output2 = contrastive_model(data_copy2)
                                        temp_output2[0, target].backward()
                                        relevance2 = data_copy2.x.grad.sum(-1)
                                for key, val in model_copy.saved_rels.items():
                                    if exp_method == 'lrp' or exp_method == 'cam':
                                        explain_output[f'{exp_method}_class{target}_{key}'] = val.tolist()
                                    elif exp_method == 'eb':  # Obtain contrastive result
                                        contrastive_val = contrastive_model.saved_rels.get(key, None)
                                        if contrastive_val is None:
                                            print('Error in contrastive values')
                                            raise Exception
                                        new_val = val - contrastive_val
                                        explain_output[f'eb_class{target}_{key}'] = new_val.tolist()
                                if exp_method == 'eb':  # Obtain contrastive result
                                    contrastive_result = (relevance * torch.exp(temp_output[0, target])) - \
                                                         (relevance2 * torch.exp(temp_output2[0, target]))
                                    relevance_all_classes[:, target] = contrastive_result.cpu()
                                    relevance_top_k = torch.topk(contrastive_result, k=contrastive_result.shape[0])
                                else:
                                    relevance_all_classes[:, target] = (
                                                relevance * torch.exp(temp_output[0, target])).cpu()
                                    relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                                # print(root_tweetid, target, relevance_top_k)
                                explain_output[f'{exp_method}_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                       relevance_top_k.values.tolist()]
                            else:
                                relevance_all_classes = torch.sum(relevance_all_classes, -1)
                                relevance_all_classes /= relevance_all_classes.sum()
                                relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                                explain_output[f'{exp_method}_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                  relevance_top_k.values.tolist()]
                            out_labels = net(data_sample)
                            # print(explain_output[f'lrp_allclass_top_k'])
                            # raise Exception
                        elif model == 'EBGCN':
                            # if version == 2:  # Version 2
                            #     new_x = x
                            #     new_x = new_x.reshape(new_x.shape[0], -1, 768)
                            #     new_x = new_x[:, 0]
                            #     x = new_x
                            #     data_sample.x = x
                            if 2 <= version <= 3:
                                try:
                                    data_sample.x = data_sample.cls
                                except:
                                    pass
                            x = data_sample.x
                            # TD
                            td_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.TDrumorGCN, x, edge_index,
                                                                                          data_sample, device)
                            explain_output['td_gcn_explanations'] = td_gcn_explanations
                            # BU
                            bu_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.BUrumorGCN, x, BU_edge_index,
                                                                                          data_sample, device)
                            explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                            # ebgcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            # ebgcn_copy.eval()
                            relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                            for target in range(4):
                                data_copy = copy.deepcopy(data_sample)
                                data_copy.x.requires_grad = True
                                model_copy.zero_grad()
                                if exp_method == 'eb':  # Extra data copy for contrastive model
                                    data_copy2 = copy.deepcopy(data_sample)
                                    data_copy2.x.requires_grad = True
                                    contrastive_model.zero_grad()
                                with torch.enable_grad():
                                    temp_output, _, _ = model_copy(data_copy)
                                    temp_output[0, target].backward()
                                    relevance = data_copy.x.grad.sum(-1)
                                    if exp_method == 'eb':  # Backprop through contrastive model
                                        temp_output2, _, _ = contrastive_model(data_copy2)
                                        temp_output2[0, target].backward()
                                        relevance2 = data_copy2.x.grad.sum(-1)
                                for key, val in model_copy.saved_rels.items():
                                    if exp_method == 'lrp' or exp_method == 'cam':
                                        explain_output[f'{exp_method}_class{target}_{key}'] = val.tolist()
                                    elif exp_method == 'eb':  # Obtain contrastive result
                                        contrastive_val = contrastive_model.saved_rels.get(key, None)
                                        if contrastive_val is None:
                                            print('Error in contrastive values')
                                            raise Exception
                                        new_val = val - contrastive_val
                                        explain_output[f'eb_class{target}_{key}'] = new_val.tolist()
                                if exp_method == 'eb':  # Obtain contrastive result
                                    contrastive_result = (relevance * torch.exp(temp_output[0, target])) - \
                                                         (relevance2 * torch.exp(temp_output2[0, target]))
                                    relevance_all_classes[:, target] = contrastive_result.cpu()
                                    relevance_top_k = torch.topk(contrastive_result, k=contrastive_result.shape[0])
                                else:
                                    relevance_all_classes[:, target] = (
                                                relevance * torch.exp(temp_output[0, target])).cpu()
                                    relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                                # print(root_tweetid, target, relevance_top_k)
                                explain_output[f'{exp_method}_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                       relevance_top_k.values.tolist()]
                            else:
                                relevance_all_classes = torch.sum(relevance_all_classes, -1)
                                relevance_all_classes /= relevance_all_classes.sum()
                                relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                                explain_output[f'{exp_method}_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                  relevance_top_k.values.tolist()]
                            out_labels, _, _ = net(data_sample)
                        _, pred = out_labels.max(dim=-1)
                        correct = pred.eq(data_sample.y).sum().item()
                        # print(pred.item(), data_sample.y.item(), correct)
                        explain_output['logits'] = out_labels.tolist()
                        explain_output['prediction'] = pred.item()
                        explain_output['ground_truth'] = data_sample.y.item()
                        explain_output['correct_prediction'] = correct
                        # Process LRP maps to reduce size
                        if exp_method == 'lrp':
                            process_lrp_contributions(explain_output)
                        elif exp_method == 'cam':
                            process_cam_contributions(explain_output)
                        elif exp_method == 'eb':
                            process_eb_contributions(explain_output)
                        if data_sample.x.shape[0] >= min_graph_size:
                            eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\n'
                        else:
                            eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\t' \
                                               f'Skipped\n'
                        conf_mat[data_sample.y.item(), pred.item()] += 1
                        if data_sample.x.shape[0] < min_graph_size:
                            num_skipped += 1
                            continue
                        fold_output[int(root_tweetid[0][0])] = explain_output
                        save = True
                        if save:
                            save_path = os.path.join(SAVE_DIR_PATH, event_name,
                                                     f'{root_tweetid[0][0]}_{model0}_d{drop_edges}_explain.json')
                            # print(f'Saving to {save_path}')
                            try:
                                with open(save_path, 'w') as f:
                                    json.dump(explain_output, f, indent=1)
                                # print(f'\rSaved to {save_path}')
                            except:
                                print(f'\rFailed to save to {save_path}')
                    outputs.append(fold_output)
                    total_evaluated = conf_mat.sum()
                    total_correct = conf_mat.diagonal().sum()
                    acc = total_correct / total_evaluated
                    eval_log_string += f'Skipped: {num_skipped / total_num * 100:.2f}% [{num_skipped}]/[{total_num}]\n'
                    eval_log_string += f'Acc: {acc * 100:.5f}% [{total_correct}]/[{total_evaluated}]\n'
                    for i in range(4):
                        precision = conf_mat[i, i] / conf_mat[:, i].sum()
                        recall = conf_mat[i, i] / conf_mat[i, :].sum()
                        f1 = 2 * precision * recall / (precision + recall)
                        eval_log_string += f'Class {i}:\t' \
                                           f'Precision: {precision}\t' \
                                           f'Recall: {recall}\t' \
                                           f'F1: {f1}\n'
                    eval_log_string += f' {"":20} | {"":20} | {"Predicted":20}\n' \
                                       f' {"":20} | {"":20} | {"Class 0":20} | {"Class 1":20} | {"Class 2":20} | {"Class 3":20}\n'
                    for i in range(4):
                        if i != 0:
                            eval_log_string += f' {"":20} | {f"Class {i}":20} |'
                        else:
                            eval_log_string += f' {"Actual":20} | {f"Class {i}":20} |'
                        eval_log_string += f' {conf_mat[i, 0]:20} | {conf_mat[i, 1]:20} |' \
                                           f' {conf_mat[i, 2]:20} | {conf_mat[i, 3]:20}\n'
                    save = True
                    if save:
                        with open(evaluation_log_path, 'w') as f:
                            f.write(eval_log_string)
    elif config_type == 'weights':
        for weight in set_initial_weight_types:
            print(f'\nGenerating:\t'
                  f'Model: {model0}\t'
                  f'Set Initial Weights: {weight}')
            if split_type == '5fold':
                for fold_num, fold in enumerate(load5foldData(datasetname)):
                    if fold_num % 2 != 0:  # Training fold, skip this
                        continue
                    else:
                        fold_num = fold_num // 2
                        # if fold_num != 0:  # MemoryError
                        #     continue
                    net = reinit_net()
                    try:
                        checkpoint_path = os.path.join(CHECKPOINT_DIR, datasetname, checkpoint_paths[fold_num])
                        checkpoint = torch.load(checkpoint_path)
                        net.load_state_dict(checkpoint['model_state_dict'])
                        print(f'Checkpoint loaded from {checkpoint_path}')
                    except:
                        print('No checkpoint to load')
                    net.eval()

                    model_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                    model_copy.eval()
                    if exp_method == 'eb':  # Copy contrastive model
                        net_copy = copy.deepcopy(net)
                        contrastive_model = lrp_utils.get_lrpwrappermodule(net_copy, LRP_PARAMS, is_contrastive=True)
                        contrastive_model.eval()
                    fold_output = {}

                    event_name = f'fold{fold_num}'
                    if not os.path.exists(os.path.join(SAVE_DIR_PATH, event_name)):
                        os.makedirs(os.path.join(SAVE_DIR_PATH, event_name))
                        print(f'Save directory for {event_name} created at: '
                              f'{os.path.join(SAVE_DIR_PATH, event_name)}\n')
                    else:
                        print(f'Save directory for {event_name} already exists at: '
                              f'{os.path.join(SAVE_DIR_PATH, event_name)}\n')
                    fold_test = fold
                    fold_train = []
                    treeDic = loadTree(datasetname)
                    traindata_list, testdata_list = loadBiData(datasetname,
                                                               treeDic,
                                                               fold_train,
                                                               fold_test,
                                                               TDdroprate,
                                                               BUdroprate)
                    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=5)
                    evaluation_log_path = os.path.join(EXPLAIN_DIR,
                                                       f'{datasetname}_{event_name}_{model0}_w{weight}_eval_'
                                                       f'{exp_method}.txt')
                    eval_log_string = ''
                    conf_mat = np.zeros((4, 4))
                    num_skipped = 0
                    total_num = 0
                    for sample_num, (data_sample, root_tweetid) in enumerate(tqdm(test_loader)):
                        total_num += 1
                        explain_output = {}
                        # print(type(data_sample['edge_index']), isinstance(data_sample['edge_index'], torch_sparse.SparseTensor))
                        # data_sample.retains_grad = True
                        data_sample = data_sample.to(device)

                        if 2 <= version <= 3:
                            try:
                                data_sample.x = data_sample.cls
                            except:
                                pass
                        x = data_sample.x
                        edge_index = data_sample.edge_index
                        BU_edge_index = data_sample.BU_edge_index
                        edge_weights = torch.ones_like(edge_index[0]).long() * weight
                        data_sample.edge_weights = edge_weights.to(device)
                        # tweetids = data_sample.tweetids
                        # explain_output['tweetids'] = tweetids.tolist()
                        #
                        # node_num_to_tweetid = {}
                        # for node_num, tweetid in enumerate(tweetids):
                        #     tweetid: torch.Tensor
                        #     node_num_to_tweetid[int(node_num)] = tweetid.item()
                        #
                        # explain_output['node_num_to_tweetid'] = node_num_to_tweetid

                        explain_output['rootindex'] = data_sample.rootindex.item()

                        # x_sum = torch.sum(data_sample.x, dim=-1)
                        # print('x_sum', x_sum)
                        # x_sum_top_k = torch.topk(x_sum,
                        #                          k=x_sum.shape[0])
                        # if x.shape[0] != len(node_num_to_tweetid) \
                        #         or x.shape[0] != x_sum.shape[0] \
                        #         or x.shape[0] != x_sum_top_k.indices.shape[0] \
                        #         or x_sum.shape[0] != x_sum_top_k.indices.shape[0]:
                        #     print('Error')
                        # print(root_tweetid, len(tweetids), x.shape, x_sum.shape, x_sum_top_k.indices.shape)
                        # raise Exception
                        # print(root_tweetid, len(node_num_to_tweetid), x.shape, x_sum.shape, x_sum_top_k.indices.shape)
                        # print('x_sum_top_k', x_sum_top_k)
                        # explain_output['x_sum_top_k'] = [x_sum_top_k.indices.tolist(),
                        #                                  x_sum_top_k.values.tolist()]

                        # TODO: Need to finish the extract method
                        if model == 'BiGCN':
                            # if version == 2:  # Version 2
                            #     new_x = x
                            #     new_x = new_x.reshape(new_x.shape[0], -1, 768)
                            #     new_x = new_x[:, 0]
                            #     x = new_x
                            #     data_sample.x = x
                            if 2 <= version <= 3:
                                try:
                                    data_sample.x = data_sample.cls
                                except:
                                    pass
                            x = data_sample.x
                            # TD
                            td_gcn_conv1 = net.TDrumorGCN.conv1
                            td_gcn_conv2 = net.TDrumorGCN.conv2
                            td_gcn_explanations = extract_intermediates_bigcn_twitter1516(td_gcn_conv1, td_gcn_conv2, x,
                                                                                          edge_index, data_sample, device)
                            explain_output['td_gcn_explanations'] = td_gcn_explanations
                            # BU
                            bu_gcn_conv1 = net.BUrumorGCN.conv1
                            bu_gcn_conv2 = net.BUrumorGCN.conv2
                            bu_gcn_explanations = extract_intermediates_bigcn_twitter1516(bu_gcn_conv1, bu_gcn_conv2, x,
                                                                                          BU_edge_index, data_sample, device)
                            explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                            # bigcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            # bigcn_copy.eval()
                            relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                            for target in range(4):
                                data_copy = copy.deepcopy(data_sample)
                                data_copy.x.requires_grad = True
                                model_copy.zero_grad()
                                if exp_method == 'eb':  # Extra data copy for contrastive model
                                    data_copy2 = copy.deepcopy(data_sample)
                                    data_copy2.x.requires_grad = True
                                    contrastive_model.zero_grad()
                                with torch.enable_grad():
                                    temp_output = model_copy(data_copy)
                                    temp_output[0, target].backward()
                                    relevance = data_copy.x.grad.sum(-1)
                                    if exp_method == 'eb':  # Backprop through contrastive model
                                        # with torch.enable_grad():
                                        temp_output2 = contrastive_model(data_copy2)
                                        temp_output2[0, target].backward()
                                        # print(temp_output2.grad)
                                        relevance2 = data_copy2.x.grad.sum(-1)
                                for key, val in model_copy.saved_rels.items():
                                    if exp_method == 'lrp' or exp_method == 'cam':
                                        explain_output[f'{exp_method}_class{target}_{key}'] = val.tolist()
                                    elif exp_method == 'eb':  # Obtain contrastive result
                                        contrastive_val = contrastive_model.saved_rels.get(key, None)
                                        if contrastive_val is None:
                                            print('Error in contrastive values')
                                            raise Exception
                                        new_val = val - contrastive_val
                                        explain_output[f'eb_class{target}_{key}'] = new_val.tolist()
                                if exp_method == 'eb':  # Obtain contrastive result
                                    contrastive_result = (relevance * torch.exp(temp_output[0, target])) - \
                                                         (relevance2 * torch.exp(temp_output2[0, target]))
                                    relevance_all_classes[:, target] = contrastive_result.cpu()
                                    relevance_top_k = torch.topk(contrastive_result, k=contrastive_result.shape[0])
                                else:
                                    relevance_all_classes[:, target] = (
                                                relevance * torch.exp(temp_output[0, target])).cpu()
                                    relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                                # print(root_tweetid, target, relevance_top_k)
                                explain_output[f'{exp_method}_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                       relevance_top_k.values.tolist()]
                            else:
                                relevance_all_classes = torch.sum(relevance_all_classes, -1)
                                relevance_all_classes /= relevance_all_classes.sum()
                                relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                                explain_output[f'{exp_method}_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                  relevance_top_k.values.tolist()]
                            out_labels = net(data_sample)
                        elif model == 'EBGCN':
                            # if version == 2:  # Version 2
                            #     new_x = x
                            #     new_x = new_x.reshape(new_x.shape[0], -1, 768)
                            #     new_x = new_x[:, 0]
                            #     x = new_x
                            #     data_sample.x = x
                            if 2 <= version <= 3:
                                try:
                                    data_sample.x = data_sample.cls
                                except:
                                    pass
                            # TD
                            td_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.TDrumorGCN, x, edge_index,
                                                                                          data_sample, device)
                            explain_output['td_gcn_explanations'] = td_gcn_explanations
                            # BU
                            bu_gcn_explanations = extract_intermediates_ebgcn_twitter1516(net.BUrumorGCN, x, BU_edge_index,
                                                                                          data_sample, device)
                            explain_output['bu_gcn_explanations'] = bu_gcn_explanations
                            # ebgcn_copy = lrp_utils.get_lrpwrappermodule(net, LRP_PARAMS)
                            # ebgcn_copy.eval()
                            relevance_all_classes = torch.zeros((data_sample.x.shape[0], 4))
                            for target in range(4):
                                data_copy = copy.deepcopy(data_sample)
                                data_copy.x.requires_grad = True
                                model_copy.zero_grad()
                                if exp_method == 'eb':  # Extra data copy for contrastive model
                                    data_copy2 = copy.deepcopy(data_sample)
                                    data_copy2.x.requires_grad = True
                                    contrastive_model.zero_grad()
                                with torch.enable_grad():
                                    temp_output, _, _ = model_copy(data_copy)
                                    temp_output[0, target].backward()
                                    relevance = data_copy.x.grad.sum(-1)
                                    if exp_method == 'eb':  # Backprop through contrastive model
                                        temp_output2, _, _ = contrastive_model(data_copy2)
                                        temp_output2[0, target].backward()
                                        relevance2 = data_copy2.x.grad.sum(-1)
                                for key, val in model_copy.saved_rels.items():
                                    if exp_method == 'lrp' or exp_method == 'cam':
                                        explain_output[f'{exp_method}_class{target}_{key}'] = val.tolist()
                                    elif exp_method == 'eb':  # Obtain contrastive result
                                        contrastive_val = contrastive_model.saved_rels.get(key, None)
                                        if contrastive_val is None:
                                            print('Error in contrastive values')
                                            raise Exception
                                        new_val = val - contrastive_val
                                        explain_output[f'eb_class{target}_{key}'] = new_val.tolist()
                                if exp_method == 'eb':  # Obtain contrastive result
                                    contrastive_result = (relevance * torch.exp(temp_output[0, target])) - \
                                                         (relevance2 * torch.exp(temp_output2[0, target]))
                                    relevance_all_classes[:, target] = contrastive_result.cpu()
                                    relevance_top_k = torch.topk(contrastive_result, k=contrastive_result.shape[0])
                                else:
                                    relevance_all_classes[:, target] = (
                                                relevance * torch.exp(temp_output[0, target])).cpu()
                                    relevance_top_k = torch.topk(relevance, k=relevance.shape[0])
                                # print(root_tweetid, target, relevance_top_k)
                                explain_output[f'{exp_method}_class{target}_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                       relevance_top_k.values.tolist()]
                            else:
                                relevance_all_classes = torch.sum(relevance_all_classes, -1)
                                relevance_all_classes /= relevance_all_classes.sum()
                                relevance_top_k = torch.topk(relevance_all_classes, k=relevance_all_classes.shape[0])
                                explain_output[f'{exp_method}_allclass_top_k'] = [relevance_top_k.indices.tolist(),
                                                                                  relevance_top_k.values.tolist()]
                            out_labels, _, _ = net(data_sample)
                        _, pred = out_labels.max(dim=-1)
                        correct = pred.eq(data_sample.y).sum().item()
                        # print(pred.item(), data_sample.y.item(), correct)
                        explain_output['logits'] = out_labels.tolist()
                        explain_output['prediction'] = pred.item()
                        explain_output['ground_truth'] = data_sample.y.item()
                        explain_output['correct_prediction'] = correct
                        # Process LRP maps to reduce size
                        if exp_method == 'lrp':
                            process_lrp_contributions(explain_output)
                        elif exp_method == 'cam':
                            process_cam_contributions(explain_output)
                        elif exp_method == 'eb':
                            process_eb_contributions(explain_output)
                        if data_sample.x.shape[0] >= min_graph_size:
                            eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\n'
                        else:
                            eval_log_string += f'{root_tweetid[0][0]}: pred: {pred.item()} gt: {data_sample.y.item()}\t' \
                                               f'Skipped\n'
                        conf_mat[data_sample.y.item(), pred.item()] += 1
                        if data_sample.x.shape[0] < min_graph_size:
                            num_skipped += 1
                            continue
                        fold_output[int(root_tweetid[0][0])] = explain_output
                        save = True
                        if save:
                            save_path = os.path.join(SAVE_DIR_PATH, event_name,
                                                     f'{root_tweetid[0][0]}_{model0}_w{weight}_explain.json')
                            # print(f'Saving to {save_path}')
                            try:
                                with open(save_path, 'w') as f:
                                    json.dump(explain_output, f, indent=1)
                                # print(f'\rSaved to {save_path}')
                            except:
                                print(f'\rFailed to save to {save_path}')
                    outputs.append(fold_output)
                    total_evaluated = conf_mat.sum()
                    total_correct = conf_mat.diagonal().sum()
                    acc = total_correct / total_evaluated
                    eval_log_string += f'Skipped: {num_skipped / total_num * 100:.2f}% [{num_skipped}]/[{total_num}]\n'
                    eval_log_string += f'Acc: {acc * 100:.5f}% [{total_correct}]/[{total_evaluated}]\n'
                    for i in range(4):
                        precision = conf_mat[i, i] / conf_mat[:, i].sum()
                        recall = conf_mat[i, i] / conf_mat[i, :].sum()
                        f1 = 2 * precision * recall / (precision + recall)
                        eval_log_string += f'Class {i}:\t' \
                                           f'Precision: {precision}\t' \
                                           f'Recall: {recall}\t' \
                                           f'F1: {f1}\n'
                    eval_log_string += f' {"":20} | {"":20} | {"Predicted":20}\n' \
                                       f' {"":20} | {"":20} | {"Class 0":20} | {"Class 1":20} | {"Class 2":20} | {"Class 3":20}\n'
                    for i in range(4):
                        if i != 0:
                            eval_log_string += f' {"":20} | {f"Class {i}":20} |'
                        else:
                            eval_log_string += f' {"Actual":20} | {f"Class {i}":20} |'
                        eval_log_string += f' {conf_mat[i, 0]:20} | {conf_mat[i, 1]:20} |' \
                                           f' {conf_mat[i, 2]:20} | {conf_mat[i, 3]:20}\n'
                    save = True
                    if save:
                        with open(evaluation_log_path, 'w') as f:
                            f.write(eval_log_string)
    print('End of programme')
