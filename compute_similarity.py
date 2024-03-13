import os
import json
import sys, traceback

import scipy.stats as stats
import scipy.special as special
import numpy as np
import torch
from tqdm import tqdm
from Process.rand5fold import load5foldDataStratified
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from lrp_pytorch.modules.base import safe_divide

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPLAIN_DIR = os.path.join(DATA_DIR, 'explain')
CENTRALITY_DIR = os.path.join(DATA_DIR, 'centrality')

EXTERNAL_DRIVE_PATH = os.path.join('D', 'BiGCN', 'data')
EXTERNAL_EXPLAIN_DIR = os.path.join(EXTERNAL_DRIVE_PATH, 'explain')

DATASETNAME = 'PHEME'
EVENT = 'charliehebdo'
FOLD_2_EVENTNAME = {0: 'charliehebdo',
                    1: 'ebola',
                    2: 'ferguson',
                    3: 'germanwings',
                    4: 'gurlitt',
                    5: 'ottawashooting',
                    6: 'prince',
                    7: 'putinmissing',
                    8: 'sydneysiege'}


def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    #Find intersection of two sets
    nominator = a.intersection(b)
    #Find union of two sets
    denominator = a.union(b)
    #Take the ratio of sizes
    try:
        similarity = len(nominator)/len(denominator)
    except:
        # print(a, b, nominator, denominator)
        raise Exception
    # print(a, b, nominator, denominator, similarity)
    return similarity


def szymkiewicz_simpson(a, b):
    a = set(a)
    b = set(b)
    nominator = a.intersection(b)
    if len(a) <= len(b):
        denominator = a
    else:
        denominator = b
    try:
        similarity = len(nominator)/len(denominator)
    except:
        raise Exception
    return similarity


def compute_graph_contributions(edge_index, edge_weights1, edge_weights2):
    edge_list = list(map(lambda x, y: (x, y), edge_index[0], edge_index[1]))
    edge_weights_dict = {f'{src}-{dst}': (edge_weights1[i], edge_weights2[i]) for i, (src, dst) in enumerate(edge_list)}
    g = nx.DiGraph(edge_list)
    edge_contributions_dict = {}
    for edge_num, edge in enumerate(edge_list):
        src, dst = edge
        if src == dst:
            continue
        edge_contributions_dict[f'{src}-{dst}'] = edge_weights1[edge_num] + edge_weights2[edge_num]
        children = nx.descendants(g, dst)
        for child in children:
            edge_contributions_dict[f'{src}-{dst}'] += edge_weights_dict.get(f'{dst}-{child}', (0, 0))[1]
    sum_edge_contributions = sum(map(lambda x: edge_contributions_dict[x], edge_contributions_dict.keys()))
    normalised_edge_contributions_dict = {k: v/sum_edge_contributions for k, v in edge_contributions_dict.items()}
    node_contributions_dict = {}
    for edge_num, edge in enumerate(edge_list):
        src, dst = edge
        if src != dst:
            continue
        node_contributions_dict[f'{src}'] = edge_weights1[edge_num] + edge_weights2[edge_num]
        children = nx.descendants(g, src)
        for child in children:
            node_contributions_dict[f'{src}'] += edge_contributions_dict.get(f'{src}-{child}', 0)
    sum_node_contributions = sum(map(lambda x: node_contributions_dict[x], node_contributions_dict.keys()))
    normalised_node_contributions_dict = {k: v / sum_node_contributions for k, v in node_contributions_dict.items()}
    return normalised_node_contributions_dict, normalised_edge_contributions_dict


def compute_text_contribution(text_act1, text_act2):
    return list(map(lambda x, y: x + y, text_act1, text_act2))


def compute_gcl_output(text_vecs, edge_index, edge_weights):
    text_vecs = np.asarray(text_vecs)
    edge_index = np.asarray(edge_index)
    edge_weights = np.asarray(edge_weights)
    output_mat = np.zeros(text_vecs.shape)
    for edge_num in range(edge_index.shape[-1]):
        src, dst = edge_index[0, edge_num], edge_index[1, edge_num]
        output_mat[dst] += text_vecs[src] * edge_weights[edge_num]
    return output_mat


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


def load_layerwise_explanations(bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log):
    try:
        # BiGCN
        with open(bigcn_explain_json_path, 'r') as f:
            bigcn_explain_json = json.load(f)
        if len(bigcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk'][0]) < min_graph_size:
            error_log.append(f'Event {tree_id} graph smaller than {min_graph_size}')
            return None
    except:
        error_log.append(f'Error: Missing BiGCN explanation files for Tree num {tree_id}\t'
                         f'{bigcn_explain_json_path}')
        return None
    # TD
    bigcn_td_conv1_text = bigcn_explain_json['td_gcn_explanations']['conv1_text'][0]
    bigcn_td_conv2_text = bigcn_explain_json['td_gcn_explanations']['conv2_text'][0]
    bigcn_td_conv1_edge_weights = bigcn_explain_json['td_gcn_explanations']['conv1_edge_weights']
    bigcn_td_conv2_edge_weights = bigcn_explain_json['td_gcn_explanations']['conv2_edge_weights']

    bigcn_lrp_class0_td_conv1 = bigcn_explain_json['lrp_class0_td_conv1']
    bigcn_lrp_class1_td_conv1 = bigcn_explain_json['lrp_class1_td_conv1']
    bigcn_lrp_class2_td_conv1 = bigcn_explain_json['lrp_class2_td_conv1']
    bigcn_lrp_class3_td_conv1 = bigcn_explain_json['lrp_class3_td_conv1']

    bigcn_lrp_class0_td_conv2 = bigcn_explain_json['lrp_class0_td_conv2']
    bigcn_lrp_class1_td_conv2 = bigcn_explain_json['lrp_class1_td_conv2']
    bigcn_lrp_class2_td_conv2 = bigcn_explain_json['lrp_class2_td_conv2']
    bigcn_lrp_class3_td_conv2 = bigcn_explain_json['lrp_class3_td_conv2']

    logits = bigcn_explain_json['logits']
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

    bigcn_lrp_class0_td_conv1_text, _ = get_text_ranked_list(bigcn_lrp_class0_td_conv1_text,
                                                             np.zeros(bigcn_lrp_class0_td_conv1_text.shape))
    bigcn_lrp_class1_td_conv1_text, _ = get_text_ranked_list(bigcn_lrp_class1_td_conv1_text,
                                                             np.zeros(bigcn_lrp_class1_td_conv1_text.shape))
    bigcn_lrp_class2_td_conv1_text, _ = get_text_ranked_list(bigcn_lrp_class2_td_conv1_text,
                                                             np.zeros(bigcn_lrp_class2_td_conv1_text.shape))
    bigcn_lrp_class3_td_conv1_text, _ = get_text_ranked_list(bigcn_lrp_class3_td_conv1_text,
                                                             np.zeros(bigcn_lrp_class3_td_conv1_text.shape))

    bigcn_lrp_class0_td_conv2_text, _ = get_text_ranked_list(bigcn_lrp_class0_td_conv2_text,
                                                             np.zeros(bigcn_lrp_class0_td_conv2_text.shape))
    bigcn_lrp_class1_td_conv2_text, _ = get_text_ranked_list(bigcn_lrp_class1_td_conv2_text,
                                                             np.zeros(bigcn_lrp_class1_td_conv2_text.shape))
    bigcn_lrp_class2_td_conv2_text, _ = get_text_ranked_list(bigcn_lrp_class2_td_conv2_text,
                                                             np.zeros(bigcn_lrp_class2_td_conv2_text.shape))
    bigcn_lrp_class3_td_conv2_text, _ = get_text_ranked_list(bigcn_lrp_class3_td_conv2_text,
                                                             np.zeros(bigcn_lrp_class3_td_conv2_text.shape))

    # bigcn_lrp_class0_td_text, t0 = get_text_ranked_list(bigcn_lrp_class0_td_conv1_text,
    #                                                     bigcn_lrp_class0_td_conv2_text)
    # bigcn_lrp_class1_td_text, t1 = get_text_ranked_list(bigcn_lrp_class1_td_conv1_text,
    #                                                     bigcn_lrp_class1_td_conv2_text)
    # bigcn_lrp_class2_td_text, t2 = get_text_ranked_list(bigcn_lrp_class1_td_conv1_text,
    #                                                     bigcn_lrp_class2_td_conv2_text)
    # bigcn_lrp_class3_td_text, t3 = get_text_ranked_list(bigcn_lrp_class1_td_conv1_text,
    #                                                     bigcn_lrp_class3_td_conv2_text)
    # bigcn_lrp_allclass_td_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
    #                              np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    # bigcn_lrp_allclass_td_text /= bigcn_lrp_allclass_td_text.sum()
    # bigcn_lrp_allclass_td_text = torch.topk(torch.as_tensor(bigcn_lrp_allclass_td_text),
    #                                         k=bigcn_lrp_allclass_td_text.shape[0])
    # bigcn_lrp_allclass_td_text = [bigcn_lrp_allclass_td_text.indices.tolist(),
    #                               bigcn_lrp_allclass_td_text.values.tolist()]
    #
    # bigcn_lrp_class0_td_edge, e0 = get_edge_ranked_list(bigcn_lrp_class0_td_conv1_edge,
    #                                                     bigcn_lrp_class0_td_conv2_edge)
    # bigcn_lrp_class1_td_edge, e1 = get_edge_ranked_list(bigcn_lrp_class1_td_conv1_edge,
    #                                                     bigcn_lrp_class1_td_conv2_edge)
    # bigcn_lrp_class2_td_edge, e2 = get_edge_ranked_list(bigcn_lrp_class1_td_conv1_edge,
    #                                                     bigcn_lrp_class2_td_conv2_edge)
    # bigcn_lrp_class3_td_edge, e3 = get_edge_ranked_list(bigcn_lrp_class1_td_conv1_edge,
    #                                                     bigcn_lrp_class3_td_conv2_edge)
    # bigcn_lrp_allclass_td_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
    #                              np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    # bigcn_lrp_allclass_td_edge /= bigcn_lrp_allclass_td_edge.sum()
    # bigcn_lrp_allclass_td_edge = torch.topk(torch.as_tensor(bigcn_lrp_allclass_td_edge),
    #                                         k=bigcn_lrp_allclass_td_edge.shape[0])
    # bigcn_lrp_allclass_td_edge = [bigcn_lrp_allclass_td_edge.indices.tolist(),
    #                               bigcn_lrp_allclass_td_edge.values.tolist()]
    # BU
    bigcn_bu_conv1_text = bigcn_explain_json['bu_gcn_explanations']['conv1_text'][0]
    bigcn_bu_conv2_text = bigcn_explain_json['bu_gcn_explanations']['conv2_text'][0]
    bigcn_bu_conv1_edge_weights = bigcn_explain_json['bu_gcn_explanations']['conv1_edge_weights']
    bigcn_bu_conv2_edge_weights = bigcn_explain_json['bu_gcn_explanations']['conv2_edge_weights']

    bigcn_lrp_class0_bu_conv1 = bigcn_explain_json['lrp_class0_bu_conv1']
    bigcn_lrp_class1_bu_conv1 = bigcn_explain_json['lrp_class1_bu_conv1']
    bigcn_lrp_class2_bu_conv1 = bigcn_explain_json['lrp_class2_bu_conv1']
    bigcn_lrp_class3_bu_conv1 = bigcn_explain_json['lrp_class3_bu_conv1']

    bigcn_lrp_class0_bu_conv2 = bigcn_explain_json['lrp_class0_bu_conv2']
    bigcn_lrp_class1_bu_conv2 = bigcn_explain_json['lrp_class1_bu_conv2']
    bigcn_lrp_class2_bu_conv2 = bigcn_explain_json['lrp_class2_bu_conv2']
    bigcn_lrp_class3_bu_conv2 = bigcn_explain_json['lrp_class3_bu_conv2']

    logits = bigcn_explain_json['logits']
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

    # bigcn_lrp_class0_bu_text, t0 = get_text_ranked_list(bigcn_lrp_class0_bu_conv1_text,
    #                                                     bigcn_lrp_class0_bu_conv2_text)
    # bigcn_lrp_class1_bu_text, t1 = get_text_ranked_list(bigcn_lrp_class1_bu_conv1_text,
    #                                                     bigcn_lrp_class1_bu_conv2_text)
    # bigcn_lrp_class2_bu_text, t2 = get_text_ranked_list(bigcn_lrp_class1_bu_conv1_text,
    #                                                     bigcn_lrp_class2_bu_conv2_text)
    # bigcn_lrp_class3_bu_text, t3 = get_text_ranked_list(bigcn_lrp_class1_bu_conv1_text,
    #                                                     bigcn_lrp_class3_bu_conv2_text)
    # bigcn_lrp_allclass_bu_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
    #                              np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
    # bigcn_lrp_allclass_bu_text /= bigcn_lrp_allclass_bu_text.sum()
    # bigcn_lrp_allclass_bu_text = torch.topk(torch.as_tensor(bigcn_lrp_allclass_bu_text),
    #                                         k=bigcn_lrp_allclass_bu_text.shape[0])
    # bigcn_lrp_allclass_bu_text = [bigcn_lrp_allclass_bu_text.indices.tolist(),
    #                               bigcn_lrp_allclass_bu_text.values.tolist()]

    # bigcn_lrp_class0_bu_edge, e0 = get_edge_ranked_list(bigcn_lrp_class0_bu_conv1_edge,
    #                                                     bigcn_lrp_class0_bu_conv2_edge)
    # bigcn_lrp_class1_bu_edge, e1 = get_edge_ranked_list(bigcn_lrp_class1_bu_conv1_edge,
    #                                                     bigcn_lrp_class1_bu_conv2_edge)
    # bigcn_lrp_class2_bu_edge, e2 = get_edge_ranked_list(bigcn_lrp_class1_bu_conv1_edge,
    #                                                     bigcn_lrp_class2_bu_conv2_edge)
    # bigcn_lrp_class3_bu_edge, e3 = get_edge_ranked_list(bigcn_lrp_class1_bu_conv1_edge,
    #                                                     bigcn_lrp_class3_bu_conv2_edge)
    # bigcn_lrp_allclass_bu_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
    #                              np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
    # bigcn_lrp_allclass_bu_edge /= bigcn_lrp_allclass_bu_edge.sum()
    # bigcn_lrp_allclass_bu_edge = torch.topk(torch.as_tensor(bigcn_lrp_allclass_bu_edge),
    #                                         k=bigcn_lrp_allclass_bu_edge.shape[0])
    # bigcn_lrp_allclass_bu_edge = [bigcn_lrp_allclass_bu_edge.indices.tolist(),
    #                               bigcn_lrp_allclass_bu_edge.values.tolist()]

    # Whole Model
    bigcn_lrp_class0 = bigcn_explain_json['lrp_class0_top_k']
    bigcn_lrp_class1 = bigcn_explain_json['lrp_class1_top_k']
    bigcn_lrp_class2 = bigcn_explain_json['lrp_class2_top_k']
    bigcn_lrp_class3 = bigcn_explain_json['lrp_class3_top_k']
    bigcn_lrp_allclass = bigcn_explain_json['lrp_allclass_top_k']

    # Get sublist  TODO: Finish this
    # bigcn_lrp_class0_td_text = get_sublist(*bigcn_lrp_class0_td_text, select, k, quantile, interval)
    # bigcn_lrp_class1_td_text = get_sublist(*bigcn_lrp_class1_td_text, select, k, quantile, interval)
    # bigcn_lrp_class2_td_text = get_sublist(*bigcn_lrp_class2_td_text, select, k, quantile, interval)
    # bigcn_lrp_class3_td_text = get_sublist(*bigcn_lrp_class3_td_text, select, k, quantile, interval)
    # bigcn_lrp_allclass_td_text = get_sublist(*bigcn_lrp_allclass_td_text, select, k, quantile, interval)
    #
    # bigcn_lrp_class0_td_edge = get_sublist(*bigcn_lrp_class0_td_edge, select, k, quantile, interval)
    # bigcn_lrp_class1_td_edge = get_sublist(*bigcn_lrp_class1_td_edge, select, k, quantile, interval)
    # bigcn_lrp_class2_td_edge = get_sublist(*bigcn_lrp_class2_td_edge, select, k, quantile, interval)
    # bigcn_lrp_class3_td_edge = get_sublist(*bigcn_lrp_class3_td_edge, select, k, quantile, interval)
    # bigcn_lrp_allclass_td_edge = get_sublist(*bigcn_lrp_allclass_td_edge, select, k, quantile, interval)
    #
    # bigcn_lrp_class0_bu_text = get_sublist(*bigcn_lrp_class0_bu_text, select, k, quantile, interval)
    # bigcn_lrp_class1_bu_text = get_sublist(*bigcn_lrp_class1_bu_text, select, k, quantile, interval)
    # bigcn_lrp_class2_bu_text = get_sublist(*bigcn_lrp_class2_bu_text, select, k, quantile, interval)
    # bigcn_lrp_class3_bu_text = get_sublist(*bigcn_lrp_class3_bu_text, select, k, quantile, interval)
    # bigcn_lrp_allclass_bu_text = get_sublist(*bigcn_lrp_allclass_bu_text, select, k, quantile, interval)
    #
    # bigcn_lrp_class0_bu_edge = get_sublist(*bigcn_lrp_class0_bu_edge, select, k, quantile, interval)
    # bigcn_lrp_class1_bu_edge = get_sublist(*bigcn_lrp_class1_bu_edge, select, k, quantile, interval)
    # bigcn_lrp_class2_bu_edge = get_sublist(*bigcn_lrp_class2_bu_edge, select, k, quantile, interval)
    # bigcn_lrp_class3_bu_edge = get_sublist(*bigcn_lrp_class3_bu_edge, select, k, quantile, interval)
    # bigcn_lrp_allclass_bu_edge = get_sublist(*bigcn_lrp_allclass_bu_edge, select, k, quantile, interval)

    bigcn_lrp_class0 = get_sublist(*bigcn_lrp_class0, select, k, quantile, interval)
    bigcn_lrp_class1 = get_sublist(*bigcn_lrp_class1, select, k, quantile, interval)
    bigcn_lrp_class2 = get_sublist(*bigcn_lrp_class2, select, k, quantile, interval)
    bigcn_lrp_class3 = get_sublist(*bigcn_lrp_class3, select, k, quantile, interval)
    bigcn_lrp_allclass = get_sublist(*bigcn_lrp_allclass, select, k, quantile, interval)


def get_node_score_dict(node_list, score_list):
    temp_dict = dict()
    for node, score in zip(node_list, score_list):
        try:
            temp_dict[score].append(node)
        except:
            temp_dict[score] = [node]
    else:
        for key in temp_dict.keys():
            temp_dict[key] = sorted(temp_dict[key])
    return temp_dict


def get_sublist(node_list, score_list, select, k, quantile, interval):
    if select == 'quantile':
        return get_quantile(node_list, score_list, quantile)
    elif select == 'interval':
        return get_interval(node_list, score_list, interval)
    elif select == 'k':
        return get_topk(node_list, score_list, k)


def get_topk(node_list, score_list, k):
    node_score_dict = get_node_score_dict(node_list, score_list)
    temp_list = []
    values_list = sorted(node_score_dict.keys(), key=lambda x: float(x), reverse=True)
    # print(node_score_dict)
    for value in values_list:
        temp_list += node_score_dict[value]
    return temp_list[:k]


def get_quantile(node_list, score_list, quantile):
    limit = np.nanquantile(score_list, quantile)
    temp_list = []
    node_score_tuples = sorted(map(lambda x, y: (x, y), node_list, score_list), key=lambda z: z[1], reverse=True)
    for node, score in node_score_tuples:
        if score >= limit:
            temp_list.append(node)
    return temp_list


def get_interval(node_list, score_list, interval):
    max_score, min_score = max(score_list), min(score_list)
    limit = (max_score - min_score) * interval
    temp_list = []
    node_score_tuples = sorted(map(lambda x, y: (x, y), node_list, score_list), key=lambda z: z[1], reverse=True)
    for node, score in node_score_tuples:
        if score >= limit:
            temp_list.append(node)
    return temp_list


def compute_similarities_model_attribution_nfold(args):
    datasetname = args.get('datasetname', 'PHEME')  # ['Twitter', 'PHEME']
    event = args.get('event', 'fold0')
    split_type = args.get('split_type', '9fold')  # ['5fold', ''9fold]
    versions = args.get('versions', [2.1, 2.1, 2.1])
    lrp_version = args.get('lrp_version', 2)
    config_type = args.get('config_type', ['original'])  # ['original', 'nodes', 'edges', 'weights', 'centrality']
    select = args.get('select', 'quantile')  # ['quantile', 'interval', 'top-k']
    quantile = args.get('quantile', 0.75)
    interval = args.get('interval', 2)  # [0-10]
    k = args.get('k', 10)
    min_graph_size = args.get('min_graph_size', 20)
    randomise_types = args.get('randomise_types', [0.25])  # [1.0, 0.75, 0.5, 0.25]
    drop_edge_types = args.get('drop_edge_types', [1])  # [1.0, 0.75, 0.5, 0.25]
    set_initial_weight_types = args.get('set_initial_weight_types', [0.5, 2.0])  # [0.5, 1.0, 2.0]
    save_to_external = args.get('save_to_external', False)
    exp_method = args.get('exp_method', 'lrp')
    centrality_subdir = os.path.join(CENTRALITY_DIR, datasetname, event)
    if save_to_external:
        # centrality_subdir = os.path.join(CENTRALITY_DIR, datasetname, event)
        explain_subdir = os.path.join(EXTERNAL_EXPLAIN_DIR, datasetname, exp_method, event)
    else:
        explain_subdir = os.path.join(EXPLAIN_DIR, datasetname, exp_method, event)
    metrics = [jaccard_similarity, szymkiewicz_simpson]
    centrality = ['td_out_degree', 'td_betweenness', 'td_closeness', 'td_farness', 'td_eigencentrality',
                  'bu_out_degree', 'bu_betweenness', 'bu_closeness', 'bu_farness', 'bu_eigencentrality']
    vec = ['bigcn_nr_td_text', 'bigcn_fr_td_text', 'bigcn_tr_td_text', 'bigcn_ur_td_text',
           'bigcn_all_td_text',
           'ebgcn_nr_td_text', 'ebgcn_fr_td_text', 'ebgcn_tr_td_text', 'ebgcn_ur_td_text',
           'ebgcn_all_td_text',
           'bigcn_nr_bu_text', 'bigcn_fr_bu_text', 'bigcn_tr_bu_text', 'bigcn_ur_bu_text',
           'bigcn_all_bu_text',
           'ebgcn_nr_bu_text', 'ebgcn_fr_bu_text', 'ebgcn_tr_bu_text', 'ebgcn_ur_bu_text',
           'ebgcn_all_bu_text',
           'bigcn_nr_td_edge', 'bigcn_fr_td_edge', 'bigcn_tr_td_edge', 'bigcn_ur_td_edge',
           'bigcn_all_td_edge',
           'ebgcn_nr_td_edge', 'ebgcn_fr_td_edge', 'ebgcn_tr_td_edge', 'ebgcn_ur_td_edge',
           'ebgcn_all_td_edge',
           'bigcn_nr_bu_edge', 'bigcn_fr_bu_edge', 'bigcn_tr_bu_edge', 'bigcn_ur_bu_edge',
           'bigcn_all_bu_edge',
           'ebgcn_nr_bu_edge', 'ebgcn_fr_bu_edge', 'ebgcn_tr_bu_edge', 'ebgcn_ur_bu_edge',
           'ebgcn_all_bu_edge',
           'bigcn_nr_joint', 'bigcn_fr_joint', 'bigcn_tr_joint', 'bigcn_ur_joint', 'bigcn_all_joint',
           'ebgcn_nr_joint', 'ebgcn_fr_joint', 'ebgcn_tr_joint', 'ebgcn_ur_joint', 'ebgcn_all_joint']
    vec2 = ['bigcn_nr_td', 'bigcn_fr_td', 'bigcn_tr_td', 'bigcn_ur_td',
            'ebgcn_nr_td', 'ebgcn_fr_td', 'ebgcn_tr_td', 'ebgcn_ur_td',
            'bigcn_nr_bu', 'bigcn_fr_bu', 'bigcn_tr_bu', 'bigcn_ur_bu',
            'ebgcn_nr_bu', 'ebgcn_fr_bu', 'ebgcn_tr_bu', 'ebgcn_ur_bu']
    vec3 = ['bigcn_nr_fc', 'bigcn_fr_fc', 'bigcn_tr_fc', 'bigcn_ur_fc',
            'ebgcn_nr_fc', 'ebgcn_fr_fc', 'ebgcn_tr_fc', 'ebgcn_ur_fc']
    vec += vec2
    # vec += centrality
    # metrics_mat = np.zeros((len(os.listdir(centrality_subdir)), len(metrics), len(vec), len(vec)))
    metrics_mat = []
    eid_list = []
    errors = 0
    errors_in_metric = [0, 0]
    error_log = []
    error_eids = set()
    results_names = [name for name in vec]
    results_names += [name for name in centrality]
    bu_td_draw_contribution_split = np.zeros((8, 3))
    drop_edge_bu_td_draw_contribution_split = np.zeros((8, 3))
    initial_weight_bu_td_draw_contribution_split = None
    for initial_weight in set_initial_weight_types:
        if initial_weight_bu_td_draw_contribution_split is None:
            initial_weight_bu_td_draw_contribution_split = [np.zeros((8, 3))]
        else:
            initial_weight_bu_td_draw_contribution_split.append(np.zeros((8, 3)))
    if len(config_type) != 0:
        for config in config_type:
            if config == 'edges':
                for drop_edges in drop_edge_types:
                    results_names += [f'd{drop_edges}_{name}' for name in vec]
            elif config == 'weights':
                for initial_weight in set_initial_weight_types:
                    results_names += [f'w{initial_weight}_{name}' for name in vec]
    else:
        # if config_type == 'edges':
        #     for drop_edges in drop_edge_types:
        #         results_names += [f'd{drop_edges}_{name}' for name in vec]
        print('Config Types missing')
        raise Exception
    total_num = 0
    total_successful = 0
    for tree_num, filename in tqdm(enumerate(sorted(os.listdir(explain_subdir)))):
        total_num += 1
        tree_id = filename.split('_')[0]
        centrality_json_path = os.path.join(centrality_subdir, f'{tree_id}_centrality.json')

        def load_centrality(centrality_json_path, tree_id, error_log):
            try:
                # Centrality
                with open(centrality_json_path, 'r') as f:
                    centrality_json = json.load(f)
            except:
                error_log.append(f'Error: Missing centrality files for Tree num {tree_id}\t'
                                 f'{centrality_json_path}')
                return None

            def compute_farness(closeness):
                # nodes, closeness_scores = closeness
                nodes = []
                farness_scores = []
                k = len(closeness[0]) - 1
                for node_num, score in zip(*closeness):
                    nodes.append(node_num)
                    try:
                        farness_scores.append(1/k/score)
                    except:
                        farness_scores.append(sys.maxsize)
                return [nodes, farness_scores]
            td_out_degree = centrality_json['td']['out_degree']
            td_betweenness = centrality_json['td']['betweenness']
            td_closeness = centrality_json['td']['closeness']
            td_eigencentrality = centrality_json['td']['eigenvector']

            bu_out_degree = centrality_json['bu']['out_degree']
            bu_betweenness = centrality_json['bu']['betweenness']
            bu_closeness = centrality_json['bu']['closeness']
            bu_eigencentrality = centrality_json['bu']['eigenvector']

            td_farness = compute_farness(td_closeness)
            bu_farness = compute_farness(bu_closeness)

            td_out_degree = get_sublist(*td_out_degree, select, k, quantile, interval)
            td_betweenness = get_sublist(*td_betweenness, select, k, quantile, interval)
            td_closeness = get_sublist(*td_closeness, select, k, quantile, interval)
            td_farness = get_sublist(*td_farness, select, k, quantile, interval)
            td_eigencentrality = get_sublist(*td_eigencentrality, select, k, quantile, interval)

            bu_out_degree = get_sublist(*bu_out_degree, select, k, quantile, interval)
            bu_betweenness = get_sublist(*bu_betweenness, select, k, quantile, interval)
            bu_closeness = get_sublist(*bu_closeness, select, k, quantile, interval)
            bu_farness = get_sublist(*bu_farness, select, k, quantile, interval)
            bu_eigencentrality = get_sublist(*bu_eigencentrality, select, k, quantile, interval)

            vec = [td_out_degree, td_betweenness, td_closeness, td_farness, td_eigencentrality,
                   bu_out_degree, bu_betweenness, bu_closeness, bu_farness, bu_eigencentrality]

            return vec

        def load_model_explanations(bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log):
            try:
                # BiGCN
                with open(bigcn_explain_json_path, 'r') as f:
                    bigcn_explain_json = json.load(f)
                if len(bigcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk'][0]) < min_graph_size:
                    error_log.append(f'Event {tree_id} graph smaller than {min_graph_size}')
                    return None
            except:
                error_log.append(f'Error: Missing BiGCN explanation files for Tree num {tree_id}\t'
                                 f'{bigcn_explain_json_path}')
                return None
            # TD
            bigcn_td_conv1_act = bigcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk']
            bigcn_td_conv1_attr = bigcn_explain_json['td_gcn_explanations']['conv1_attribution_top_k']
            bigcn_td_conv2_act = bigcn_explain_json['td_gcn_explanations']['conv2_output_sum_topk']
            bigcn_td_conv2_attr = bigcn_explain_json['td_gcn_explanations']['conv2_attribution_top_k']

            bigcn_td_conv1_text = np.asarray(bigcn_explain_json['td_gcn_explanations']['conv1_text'][0]).sum(
                -1).tolist()
            bigcn_td_conv2_text = np.asarray(bigcn_explain_json['td_gcn_explanations']['conv2_text'][0]).sum(
                -1).tolist()
            bigcn_td_conv1_edge_weights = bigcn_explain_json['td_gcn_explanations']['conv1_edge_weights']
            bigcn_td_conv2_edge_weights = bigcn_explain_json['td_gcn_explanations']['conv2_edge_weights']

            bigcn_td_text = compute_text_contribution(text_act1=bigcn_td_conv1_text, text_act2=bigcn_td_conv2_text)
            bigcn_td_text_top_k = torch.topk(torch.as_tensor(bigcn_td_text), k=len(bigcn_td_text))
            bigcn_td_text = [bigcn_td_text_top_k.indices.tolist(),
                             bigcn_td_text_top_k.values.tolist()]
            bigcn_td_graph = compute_graph_contributions(edge_index=bigcn_td_conv1_edge_weights[0],
                                                         edge_weights1=bigcn_td_conv1_edge_weights[1],
                                                         edge_weights2=bigcn_td_conv2_edge_weights[1])
            bigcn_td_node, bigcn_td_edge = bigcn_td_graph
            bigcn_td_node_sl = []
            for key, val in sorted(bigcn_td_node.items()):
                bigcn_td_node_sl.append(val)
            bigcn_td_node_top_k = torch.topk(torch.as_tensor(bigcn_td_node_sl), k=len(bigcn_td_node_sl))
            bigcn_td_node = [bigcn_td_node_top_k.indices.tolist(),
                             bigcn_td_node_top_k.values.tolist()]
            # BU
            bigcn_bu_conv1_act = bigcn_explain_json['bu_gcn_explanations']['conv1_output_sum_topk']
            bigcn_bu_conv1_attr = bigcn_explain_json['bu_gcn_explanations']['conv1_attribution_top_k']
            bigcn_bu_conv2_act = bigcn_explain_json['bu_gcn_explanations']['conv2_output_sum_topk']
            bigcn_bu_conv2_attr = bigcn_explain_json['bu_gcn_explanations']['conv2_attribution_top_k']

            bigcn_bu_conv1_text = np.asarray(bigcn_explain_json['bu_gcn_explanations']['conv1_text'][0]).sum(
                -1).tolist()
            bigcn_bu_conv2_text = np.asarray(bigcn_explain_json['bu_gcn_explanations']['conv2_text'][0]).sum(
                -1).tolist()
            bigcn_bu_conv1_edge_weights = bigcn_explain_json['bu_gcn_explanations']['conv1_edge_weights']
            bigcn_bu_conv2_edge_weights = bigcn_explain_json['bu_gcn_explanations']['conv2_edge_weights']

            bigcn_bu_text = compute_text_contribution(text_act1=bigcn_bu_conv1_text, text_act2=bigcn_bu_conv2_text)
            bigcn_bu_text_top_k = torch.topk(torch.as_tensor(bigcn_bu_text), k=len(bigcn_bu_text))
            bigcn_bu_text = [bigcn_bu_text_top_k.indices.tolist(),
                             bigcn_bu_text_top_k.values.tolist()]
            bigcn_bu_graph = compute_graph_contributions(edge_index=bigcn_bu_conv1_edge_weights[0],
                                                         edge_weights1=bigcn_bu_conv1_edge_weights[1],
                                                         edge_weights2=bigcn_bu_conv2_edge_weights[1])
            bigcn_bu_node, bigcn_bu_edge = bigcn_bu_graph
            bigcn_bu_node_sl = []
            for key, val in sorted(bigcn_bu_node.items()):
                bigcn_bu_node_sl.append(val)
            bigcn_bu_node_top_k = torch.topk(torch.as_tensor(bigcn_bu_node_sl), k=len(bigcn_bu_node_sl))
            bigcn_bu_node = [bigcn_bu_node_top_k.indices.tolist(),
                             bigcn_bu_node_top_k.values.tolist()]

            bigcn_lrp_class0 = bigcn_explain_json['lrp_class0_top_k']
            bigcn_lrp_class1 = bigcn_explain_json['lrp_class1_top_k']
            bigcn_lrp_class2 = bigcn_explain_json['lrp_class2_top_k']
            bigcn_lrp_class3 = bigcn_explain_json['lrp_class3_top_k']
            bigcn_lrp_allclass = bigcn_explain_json['lrp_allclass_top_k']

            # Get sublist
            bigcn_td_conv1_act = get_sublist(*bigcn_td_conv1_act, select, k, quantile, interval)
            bigcn_td_conv1_attr = get_sublist(*bigcn_td_conv1_attr, select, k, quantile, interval)
            bigcn_td_conv2_act = get_sublist(*bigcn_td_conv2_act, select, k, quantile, interval)
            bigcn_td_conv2_attr = get_sublist(*bigcn_td_conv2_attr, select, k, quantile, interval)
            bigcn_td_text = get_sublist(*bigcn_td_text, select, k, quantile, interval)
            bigcn_td_graph = get_sublist(*bigcn_td_node, select, k, quantile, interval)

            bigcn_bu_conv1_act = get_sublist(*bigcn_bu_conv1_act, select, k, quantile, interval)
            bigcn_bu_conv1_attr = get_sublist(*bigcn_bu_conv1_attr, select, k, quantile, interval)
            bigcn_bu_conv2_act = get_sublist(*bigcn_bu_conv2_act, select, k, quantile, interval)
            bigcn_bu_conv2_attr = get_sublist(*bigcn_bu_conv2_attr, select, k, quantile, interval)
            bigcn_bu_text = get_sublist(*bigcn_bu_text, select, k, quantile, interval)
            bigcn_bu_graph = get_sublist(*bigcn_bu_node, select, k, quantile, interval)

            bigcn_lrp_class0 = get_sublist(*bigcn_lrp_class0, select, k, quantile, interval)
            bigcn_lrp_class1 = get_sublist(*bigcn_lrp_class1, select, k, quantile, interval)
            bigcn_lrp_class2 = get_sublist(*bigcn_lrp_class2, select, k, quantile, interval)
            bigcn_lrp_class3 = get_sublist(*bigcn_lrp_class3, select, k, quantile, interval)
            bigcn_lrp_allclass = get_sublist(*bigcn_lrp_allclass, select, k, quantile, interval)
            try:
                # EBGCN
                with open(ebgcn_explain_json_path, 'r') as f:
                    ebgcn_explain_json = json.load(f)
                if len(ebgcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk'][0]) < min_graph_size:
                    error_log.append(f'Event {tree_id} graph smaller than {min_graph_size}')
                    return None
            except:
                error_log.append(f'Error: Missing EBGCN explanation files for Tree num {tree_id}\t'
                                 f'{ebgcn_explain_json_path}')
                return None
            ebgcn_td_conv1_act = ebgcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk']
            ebgcn_td_conv1_attr = ebgcn_explain_json['td_gcn_explanations']['conv1_attribution_top_k']
            ebgcn_td_conv2_act = ebgcn_explain_json['td_gcn_explanations']['conv2_output_sum_topk']
            ebgcn_td_conv2_attr = ebgcn_explain_json['td_gcn_explanations']['conv2_attribution_top_k']

            ebgcn_td_conv1_text = np.asarray(ebgcn_explain_json['td_gcn_explanations']['conv1_text'][0]).sum(
                -1).tolist()
            ebgcn_td_conv2_text = np.asarray(ebgcn_explain_json['td_gcn_explanations']['conv2_text'][0]).sum(
                -1).tolist()
            ebgcn_td_conv1_edge_weights = ebgcn_explain_json['td_gcn_explanations']['conv1_edge_weights']
            ebgcn_td_conv2_edge_weights = ebgcn_explain_json['td_gcn_explanations']['conv2_edge_weights']

            ebgcn_td_text = compute_text_contribution(text_act1=ebgcn_td_conv1_text, text_act2=ebgcn_td_conv2_text)
            ebgcn_td_text_top_k = torch.topk(torch.as_tensor(ebgcn_td_text), k=len(ebgcn_td_text))
            ebgcn_td_text = [ebgcn_td_text_top_k.indices.tolist(),
                             ebgcn_td_text_top_k.values.tolist()]
            ebgcn_td_graph = compute_graph_contributions(edge_index=ebgcn_td_conv1_edge_weights[0],
                                                         edge_weights1=ebgcn_td_conv1_edge_weights[1],
                                                         edge_weights2=ebgcn_td_conv2_edge_weights[1])
            ebgcn_td_node, ebgcn_td_edge = ebgcn_td_graph
            ebgcn_td_node_sl = []
            for key, val in sorted(ebgcn_td_node.items()):
                ebgcn_td_node_sl.append(val)
            ebgcn_td_node_top_k = torch.topk(torch.as_tensor(ebgcn_td_node_sl), k=len(ebgcn_td_node_sl))
            ebgcn_td_node = [ebgcn_td_node_top_k.indices.tolist(),
                             ebgcn_td_node_top_k.values.tolist()]

            ebgcn_bu_conv1_act = ebgcn_explain_json['bu_gcn_explanations']['conv1_output_sum_topk']
            ebgcn_bu_conv1_attr = ebgcn_explain_json['bu_gcn_explanations']['conv1_attribution_top_k']
            ebgcn_bu_conv2_act = ebgcn_explain_json['bu_gcn_explanations']['conv2_output_sum_topk']
            ebgcn_bu_conv2_attr = ebgcn_explain_json['bu_gcn_explanations']['conv2_attribution_top_k']

            ebgcn_bu_conv1_text = np.asarray(ebgcn_explain_json['bu_gcn_explanations']['conv1_text'][0]).sum(
                -1).tolist()
            ebgcn_bu_conv2_text = np.asarray(ebgcn_explain_json['bu_gcn_explanations']['conv2_text'][0]).sum(
                -1).tolist()
            ebgcn_bu_conv1_edge_weights = ebgcn_explain_json['bu_gcn_explanations']['conv1_edge_weights']
            ebgcn_bu_conv2_edge_weights = ebgcn_explain_json['bu_gcn_explanations']['conv2_edge_weights']

            ebgcn_bu_text = compute_text_contribution(text_act1=ebgcn_bu_conv1_text, text_act2=ebgcn_bu_conv2_text)
            ebgcn_bu_text_top_k = torch.topk(torch.as_tensor(ebgcn_bu_text), k=len(ebgcn_bu_text))
            ebgcn_bu_text = [ebgcn_bu_text_top_k.indices.tolist(),
                             ebgcn_bu_text_top_k.values.tolist()]
            ebgcn_bu_graph = compute_graph_contributions(edge_index=ebgcn_bu_conv1_edge_weights[0],
                                                         edge_weights1=ebgcn_bu_conv1_edge_weights[1],
                                                         edge_weights2=ebgcn_bu_conv2_edge_weights[1])
            ebgcn_bu_node, ebgcn_bu_edge = ebgcn_bu_graph
            ebgcn_bu_node_sl = []
            for key, val in sorted(ebgcn_bu_node.items()):
                ebgcn_bu_node_sl.append(val)
            ebgcn_bu_node_top_k = torch.topk(torch.as_tensor(ebgcn_bu_node_sl), k=len(ebgcn_bu_node_sl))
            ebgcn_bu_node = [ebgcn_bu_node_top_k.indices.tolist(),
                             ebgcn_bu_node_top_k.values.tolist()]

            ebgcn_lrp_class0 = ebgcn_explain_json['lrp_class0_top_k']
            ebgcn_lrp_class1 = ebgcn_explain_json['lrp_class1_top_k']
            ebgcn_lrp_class2 = ebgcn_explain_json['lrp_class2_top_k']
            ebgcn_lrp_class3 = ebgcn_explain_json['lrp_class3_top_k']
            ebgcn_lrp_allclass = ebgcn_explain_json['lrp_allclass_top_k']

            # Get sublist
            ebgcn_td_conv1_act = get_sublist(*ebgcn_td_conv1_act, select, k, quantile, interval)
            ebgcn_td_conv1_attr = get_sublist(*ebgcn_td_conv1_attr, select, k, quantile, interval)
            ebgcn_td_conv2_act = get_sublist(*ebgcn_td_conv2_act, select, k, quantile, interval)
            ebgcn_td_conv2_attr = get_sublist(*ebgcn_td_conv2_attr, select, k, quantile, interval)
            ebgcn_td_text = get_sublist(*ebgcn_td_text, select, k, quantile, interval)
            ebgcn_td_graph = get_sublist(*ebgcn_td_node, select, k, quantile, interval)

            ebgcn_bu_conv1_act = get_sublist(*ebgcn_bu_conv1_act, select, k, quantile, interval)
            ebgcn_bu_conv1_attr = get_sublist(*ebgcn_bu_conv1_attr, select, k, quantile, interval)
            ebgcn_bu_conv2_act = get_sublist(*ebgcn_bu_conv2_act, select, k, quantile, interval)
            ebgcn_bu_conv2_attr = get_sublist(*ebgcn_bu_conv2_attr, select, k, quantile, interval)
            ebgcn_bu_text = get_sublist(*ebgcn_bu_text, select, k, quantile, interval)
            ebgcn_bu_graph = get_sublist(*ebgcn_bu_node, select, k, quantile, interval)

            ebgcn_lrp_class0 = get_sublist(*ebgcn_lrp_class0, select, k, quantile, interval)
            ebgcn_lrp_class1 = get_sublist(*ebgcn_lrp_class1, select, k, quantile, interval)
            ebgcn_lrp_class2 = get_sublist(*ebgcn_lrp_class2, select, k, quantile, interval)
            ebgcn_lrp_class3 = get_sublist(*ebgcn_lrp_class3, select, k, quantile, interval)
            ebgcn_lrp_allclass = get_sublist(*ebgcn_lrp_allclass, select, k, quantile, interval)

            vec = [bigcn_td_conv1_act, bigcn_td_conv1_attr, bigcn_td_conv2_act, bigcn_td_conv2_attr,
                   bigcn_td_text, bigcn_td_graph,
                   bigcn_bu_conv1_act, bigcn_bu_conv1_attr, bigcn_bu_conv2_act, bigcn_bu_conv2_attr,
                   bigcn_bu_text, bigcn_bu_graph,
                   ebgcn_td_conv1_act, ebgcn_td_conv1_attr, ebgcn_td_conv2_act, ebgcn_td_conv2_attr,
                   ebgcn_td_text, ebgcn_td_graph,
                   ebgcn_bu_conv1_act, ebgcn_bu_conv1_attr, ebgcn_bu_conv2_act, ebgcn_bu_conv2_attr,
                   ebgcn_bu_text, ebgcn_bu_graph,
                   bigcn_lrp_class0, bigcn_lrp_class1, bigcn_lrp_class2, bigcn_lrp_class3, bigcn_lrp_allclass,
                   ebgcn_lrp_class0, ebgcn_lrp_class1, ebgcn_lrp_class2, ebgcn_lrp_class3, ebgcn_lrp_allclass]
            return vec

        def load_model_explanations2(bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log):
            try:
                # BiGCN
                with open(bigcn_explain_json_path, 'r') as f:
                    bigcn_explain_json = json.load(f)
                if len(bigcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk'][0]) < min_graph_size:
                    error_log.append(f'Event {tree_id} graph smaller than {min_graph_size}')
                    return None
            except:
                error_log.append(f'Error: Missing BiGCN explanation files for Tree num {tree_id}\t'
                                 f'{bigcn_explain_json_path}')
                return None

            # TD
            bigcn_td_conv1_text = bigcn_explain_json['td_gcn_explanations']['conv1_text'][0]
            bigcn_td_conv2_text = bigcn_explain_json['td_gcn_explanations']['conv2_text'][0]
            bigcn_td_conv1_edge_weights = bigcn_explain_json['td_gcn_explanations']['conv1_edge_weights']
            bigcn_td_conv2_edge_weights = bigcn_explain_json['td_gcn_explanations']['conv2_edge_weights']

            bigcn_lrp_class0_td_conv1 = bigcn_explain_json['lrp_class0_td_conv1']
            bigcn_lrp_class1_td_conv1 = bigcn_explain_json['lrp_class1_td_conv1']
            bigcn_lrp_class2_td_conv1 = bigcn_explain_json['lrp_class2_td_conv1']
            bigcn_lrp_class3_td_conv1 = bigcn_explain_json['lrp_class3_td_conv1']

            bigcn_lrp_class0_td_conv2 = bigcn_explain_json['lrp_class0_td_conv2']
            bigcn_lrp_class1_td_conv2 = bigcn_explain_json['lrp_class1_td_conv2']
            bigcn_lrp_class2_td_conv2 = bigcn_explain_json['lrp_class2_td_conv2']
            bigcn_lrp_class3_td_conv2 = bigcn_explain_json['lrp_class3_td_conv2']

            logits = bigcn_explain_json['logits']
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
            bigcn_bu_conv1_text = bigcn_explain_json['bu_gcn_explanations']['conv1_text'][0]
            bigcn_bu_conv2_text = bigcn_explain_json['bu_gcn_explanations']['conv2_text'][0]
            bigcn_bu_conv1_edge_weights = bigcn_explain_json['bu_gcn_explanations']['conv1_edge_weights']
            bigcn_bu_conv2_edge_weights = bigcn_explain_json['bu_gcn_explanations']['conv2_edge_weights']

            bigcn_lrp_class0_bu_conv1 = bigcn_explain_json['lrp_class0_bu_conv1']
            bigcn_lrp_class1_bu_conv1 = bigcn_explain_json['lrp_class1_bu_conv1']
            bigcn_lrp_class2_bu_conv1 = bigcn_explain_json['lrp_class2_bu_conv1']
            bigcn_lrp_class3_bu_conv1 = bigcn_explain_json['lrp_class3_bu_conv1']

            bigcn_lrp_class0_bu_conv2 = bigcn_explain_json['lrp_class0_bu_conv2']
            bigcn_lrp_class1_bu_conv2 = bigcn_explain_json['lrp_class1_bu_conv2']
            bigcn_lrp_class2_bu_conv2 = bigcn_explain_json['lrp_class2_bu_conv2']
            bigcn_lrp_class3_bu_conv2 = bigcn_explain_json['lrp_class3_bu_conv2']

            logits = bigcn_explain_json['logits']
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

            # Whole Model
            bigcn_lrp_class0 = bigcn_explain_json['lrp_class0_top_k']
            bigcn_lrp_class1 = bigcn_explain_json['lrp_class1_top_k']
            bigcn_lrp_class2 = bigcn_explain_json['lrp_class2_top_k']
            bigcn_lrp_class3 = bigcn_explain_json['lrp_class3_top_k']
            bigcn_lrp_allclass = bigcn_explain_json['lrp_allclass_top_k']

            # Extra
            bigcn_nr_fc = np.asarray(bigcn_explain_json['lrp_class0_fc'])
            bigcn_fr_fc = np.asarray(bigcn_explain_json['lrp_class1_fc'])
            bigcn_tr_fc = np.asarray(bigcn_explain_json['lrp_class2_fc'])
            bigcn_ur_fc = np.asarray(bigcn_explain_json['lrp_class3_fc'])
            # print(bigcn_nr_fc.shape, bigcn_nr_fc.sum())
            # print(bigcn_nr_fc[:128].sum(), bigcn_nr_fc[128:].sum())
            # print(bigcn_fr_fc.shape, bigcn_fr_fc.sum())
            # print(bigcn_fr_fc[:128].sum(), bigcn_fr_fc[128:].sum())
            # print(bigcn_tr_fc.shape, bigcn_tr_fc.sum())
            # print(bigcn_tr_fc[:128].sum(), bigcn_tr_fc[128:].sum())
            # print(bigcn_ur_fc.shape, bigcn_ur_fc.sum())
            # print(bigcn_ur_fc[:128].sum(), bigcn_ur_fc[128:].sum())
            bigcn_nr_fc = [bigcn_nr_fc[:bigcn_nr_fc.shape[-1]//2].sum(),
                           bigcn_nr_fc[bigcn_nr_fc.shape[-1]//2:].sum()]
            bigcn_fr_fc = [bigcn_fr_fc[:bigcn_fr_fc.shape[-1]//2].sum(),
                           bigcn_fr_fc[bigcn_fr_fc.shape[-1]//2:].sum()]
            bigcn_tr_fc = [bigcn_tr_fc[:bigcn_tr_fc.shape[-1]//2].sum(),
                           bigcn_tr_fc[bigcn_tr_fc.shape[-1]//2:].sum()]
            bigcn_ur_fc = [bigcn_ur_fc[:bigcn_ur_fc.shape[-1]//2].sum(),
                           bigcn_ur_fc[bigcn_ur_fc.shape[-1]//2:].sum()]
            # print(bigcn_nr_fc)
            # print(bigcn_fr_fc)
            # print(bigcn_tr_fc)
            # print(bigcn_ur_fc)

            bigcn_nr_td_fc = np.asarray(bigcn_explain_json['lrp_class0_td_conv1_fc'])
            bigcn_fr_td_fc = np.asarray(bigcn_explain_json['lrp_class1_td_conv1_fc'])
            bigcn_tr_td_fc = np.asarray(bigcn_explain_json['lrp_class2_td_conv1_fc'])
            bigcn_ur_td_fc = np.asarray(bigcn_explain_json['lrp_class3_td_conv1_fc'])
            bigcn_nr_td_fc = [list(range(bigcn_nr_td_fc.shape[0])), bigcn_nr_td_fc.sum(-1)]
            bigcn_fr_td_fc = [list(range(bigcn_fr_td_fc.shape[0])), bigcn_fr_td_fc.sum(-1)]
            bigcn_tr_td_fc = [list(range(bigcn_tr_td_fc.shape[0])), bigcn_tr_td_fc.sum(-1)]
            bigcn_ur_td_fc = [list(range(bigcn_ur_td_fc.shape[0])), bigcn_ur_td_fc.sum(-1)]
            # print(bigcn_nr_td_fc.shape, bigcn_nr_td_fc.sum(-1))
            # print(bigcn_fr_td_fc.shape, bigcn_fr_td_fc.sum(-1))
            # print(bigcn_tr_td_fc.shape, bigcn_tr_td_fc.sum(-1))
            # print(bigcn_ur_td_fc.shape, bigcn_ur_td_fc.sum(-1))

            bigcn_nr_bu_fc = np.asarray(bigcn_explain_json['lrp_class0_bu_conv1_fc'])
            bigcn_fr_bu_fc = np.asarray(bigcn_explain_json['lrp_class1_bu_conv1_fc'])
            bigcn_tr_bu_fc = np.asarray(bigcn_explain_json['lrp_class2_bu_conv1_fc'])
            bigcn_ur_bu_fc = np.asarray(bigcn_explain_json['lrp_class3_bu_conv1_fc'])
            bigcn_nr_bu_fc = [list(range(bigcn_nr_bu_fc.shape[0])), bigcn_nr_bu_fc.sum(-1)]
            bigcn_fr_bu_fc = [list(range(bigcn_fr_bu_fc.shape[0])), bigcn_fr_bu_fc.sum(-1)]
            bigcn_tr_bu_fc = [list(range(bigcn_tr_bu_fc.shape[0])), bigcn_tr_bu_fc.sum(-1)]
            bigcn_ur_bu_fc = [list(range(bigcn_ur_bu_fc.shape[0])), bigcn_ur_bu_fc.sum(-1)]
            # print(bigcn_nr_bu_fc.shape, bigcn_nr_bu_fc.sum(-1))
            # print(bigcn_fr_bu_fc.shape, bigcn_fr_bu_fc.sum(-1))
            # print(bigcn_tr_bu_fc.shape, bigcn_tr_bu_fc.sum(-1))
            # print(bigcn_ur_bu_fc.shape, bigcn_ur_bu_fc.sum(-1))

            # Get sublist
            bigcn_lrp_class0_td_text = get_sublist(*bigcn_lrp_class0_td_text, select, k, quantile, interval)
            bigcn_lrp_class1_td_text = get_sublist(*bigcn_lrp_class1_td_text, select, k, quantile, interval)
            bigcn_lrp_class2_td_text = get_sublist(*bigcn_lrp_class2_td_text, select, k, quantile, interval)
            bigcn_lrp_class3_td_text = get_sublist(*bigcn_lrp_class3_td_text, select, k, quantile, interval)
            bigcn_lrp_allclass_td_text = get_sublist(*bigcn_lrp_allclass_td_text, select, k, quantile, interval)

            bigcn_lrp_class0_td_edge = get_sublist(*bigcn_lrp_class0_td_edge, select, k, quantile, interval)
            bigcn_lrp_class1_td_edge = get_sublist(*bigcn_lrp_class1_td_edge, select, k, quantile, interval)
            bigcn_lrp_class2_td_edge = get_sublist(*bigcn_lrp_class2_td_edge, select, k, quantile, interval)
            bigcn_lrp_class3_td_edge = get_sublist(*bigcn_lrp_class3_td_edge, select, k, quantile, interval)
            bigcn_lrp_allclass_td_edge = get_sublist(*bigcn_lrp_allclass_td_edge, select, k, quantile, interval)

            bigcn_lrp_class0_bu_text = get_sublist(*bigcn_lrp_class0_bu_text, select, k, quantile, interval)
            bigcn_lrp_class1_bu_text = get_sublist(*bigcn_lrp_class1_bu_text, select, k, quantile, interval)
            bigcn_lrp_class2_bu_text = get_sublist(*bigcn_lrp_class2_bu_text, select, k, quantile, interval)
            bigcn_lrp_class3_bu_text = get_sublist(*bigcn_lrp_class3_bu_text, select, k, quantile, interval)
            bigcn_lrp_allclass_bu_text = get_sublist(*bigcn_lrp_allclass_bu_text, select, k, quantile, interval)

            bigcn_lrp_class0_bu_edge = get_sublist(*bigcn_lrp_class0_bu_edge, select, k, quantile, interval)
            bigcn_lrp_class1_bu_edge = get_sublist(*bigcn_lrp_class1_bu_edge, select, k, quantile, interval)
            bigcn_lrp_class2_bu_edge = get_sublist(*bigcn_lrp_class2_bu_edge, select, k, quantile, interval)
            bigcn_lrp_class3_bu_edge = get_sublist(*bigcn_lrp_class3_bu_edge, select, k, quantile, interval)
            bigcn_lrp_allclass_bu_edge = get_sublist(*bigcn_lrp_allclass_bu_edge, select, k, quantile, interval)

            bigcn_lrp_class0 = get_sublist(*bigcn_lrp_class0, select, k, quantile, interval)
            bigcn_lrp_class1 = get_sublist(*bigcn_lrp_class1, select, k, quantile, interval)
            bigcn_lrp_class2 = get_sublist(*bigcn_lrp_class2, select, k, quantile, interval)
            bigcn_lrp_class3 = get_sublist(*bigcn_lrp_class3, select, k, quantile, interval)
            bigcn_lrp_allclass = get_sublist(*bigcn_lrp_allclass, select, k, quantile, interval)

            bigcn_nr_td_fc = get_sublist(*bigcn_nr_td_fc, select, k, quantile, interval)
            bigcn_fr_td_fc = get_sublist(*bigcn_fr_td_fc, select, k, quantile, interval)
            bigcn_tr_td_fc = get_sublist(*bigcn_tr_td_fc, select, k, quantile, interval)
            bigcn_ur_td_fc = get_sublist(*bigcn_ur_td_fc, select, k, quantile, interval)

            bigcn_nr_bu_fc = get_sublist(*bigcn_nr_bu_fc, select, k, quantile, interval)
            bigcn_fr_bu_fc = get_sublist(*bigcn_fr_bu_fc, select, k, quantile, interval)
            bigcn_tr_bu_fc = get_sublist(*bigcn_tr_bu_fc, select, k, quantile, interval)
            bigcn_ur_bu_fc = get_sublist(*bigcn_ur_bu_fc, select, k, quantile, interval)

            try:
                # EBGCN
                with open(ebgcn_explain_json_path, 'r') as f:
                    ebgcn_explain_json = json.load(f)
                if len(ebgcn_explain_json['td_gcn_explanations']['conv1_output_sum_topk'][0]) < min_graph_size:
                    error_log.append(f'Event {tree_id} graph smaller than {min_graph_size}')
                    return None
            except:
                error_log.append(f'Error: Missing EBGCN explanation files for Tree num {tree_id}\t'
                                 f'{ebgcn_explain_json_path}')
                return None
            ebgcn_td_conv1_text = ebgcn_explain_json['td_gcn_explanations']['conv1_text'][0]
            ebgcn_td_conv2_text = ebgcn_explain_json['td_gcn_explanations']['conv2_text'][0]
            ebgcn_td_conv1_edge_weights = ebgcn_explain_json['td_gcn_explanations']['conv1_edge_weights']
            ebgcn_td_conv2_edge_weights = ebgcn_explain_json['td_gcn_explanations']['conv2_edge_weights']

            ebgcn_lrp_class0_td_conv1 = ebgcn_explain_json['lrp_class0_td_conv1']
            ebgcn_lrp_class1_td_conv1 = ebgcn_explain_json['lrp_class1_td_conv1']
            ebgcn_lrp_class2_td_conv1 = ebgcn_explain_json['lrp_class2_td_conv1']
            ebgcn_lrp_class3_td_conv1 = ebgcn_explain_json['lrp_class3_td_conv1']

            ebgcn_lrp_class0_td_conv2 = ebgcn_explain_json['lrp_class0_td_conv2']
            ebgcn_lrp_class1_td_conv2 = ebgcn_explain_json['lrp_class1_td_conv2']
            ebgcn_lrp_class2_td_conv2 = ebgcn_explain_json['lrp_class2_td_conv2']
            ebgcn_lrp_class3_td_conv2 = ebgcn_explain_json['lrp_class3_td_conv2']

            logits = ebgcn_explain_json['logits']
            logit_weights = np.exp(np.asarray(logits))[0]

            # Conv 1
            ebgcn_lrp_class0_td_conv1_text, ebgcn_lrp_class0_td_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv1_text,
                *ebgcn_td_conv1_edge_weights, ebgcn_lrp_class0_td_conv1)
            ebgcn_lrp_class1_td_conv1_text, ebgcn_lrp_class1_td_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv1_text,
                *ebgcn_td_conv1_edge_weights, ebgcn_lrp_class1_td_conv1)
            ebgcn_lrp_class2_td_conv1_text, ebgcn_lrp_class2_td_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv1_text,
                *ebgcn_td_conv1_edge_weights, ebgcn_lrp_class2_td_conv1)
            ebgcn_lrp_class3_td_conv1_text, ebgcn_lrp_class3_td_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv1_text,
                *ebgcn_td_conv1_edge_weights, ebgcn_lrp_class3_td_conv1)

            # Conv 2
            ebgcn_lrp_class0_td_conv2_text, ebgcn_lrp_class0_td_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv2_text,
                *ebgcn_td_conv2_edge_weights, ebgcn_lrp_class0_td_conv2)
            ebgcn_lrp_class1_td_conv2_text, ebgcn_lrp_class1_td_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv2_text,
                *ebgcn_td_conv2_edge_weights, ebgcn_lrp_class1_td_conv2)
            ebgcn_lrp_class2_td_conv2_text, ebgcn_lrp_class2_td_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv2_text,
                *ebgcn_td_conv2_edge_weights, ebgcn_lrp_class2_td_conv2)
            ebgcn_lrp_class3_td_conv2_text, ebgcn_lrp_class3_td_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_td_conv2_text,
                *ebgcn_td_conv2_edge_weights, ebgcn_lrp_class3_td_conv2)

            ebgcn_lrp_class0_td_text, t0 = get_text_ranked_list(ebgcn_lrp_class0_td_conv1_text,
                                                                ebgcn_lrp_class0_td_conv2_text)
            ebgcn_lrp_class1_td_text, t1 = get_text_ranked_list(ebgcn_lrp_class1_td_conv1_text,
                                                                ebgcn_lrp_class1_td_conv2_text)
            ebgcn_lrp_class2_td_text, t2 = get_text_ranked_list(ebgcn_lrp_class1_td_conv1_text,
                                                                ebgcn_lrp_class2_td_conv2_text)
            ebgcn_lrp_class3_td_text, t3 = get_text_ranked_list(ebgcn_lrp_class1_td_conv1_text,
                                                                ebgcn_lrp_class3_td_conv2_text)
            ebgcn_lrp_allclass_td_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                         np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
            ebgcn_lrp_allclass_td_text /= ebgcn_lrp_allclass_td_text.sum()
            ebgcn_lrp_allclass_td_text = torch.topk(torch.as_tensor(ebgcn_lrp_allclass_td_text),
                                                    k=ebgcn_lrp_allclass_td_text.shape[0])
            ebgcn_lrp_allclass_td_text = [ebgcn_lrp_allclass_td_text.indices.tolist(),
                                          ebgcn_lrp_allclass_td_text.values.tolist()]

            ebgcn_lrp_class0_td_edge, e0 = get_edge_ranked_list(ebgcn_lrp_class0_td_conv1_edge,
                                                                ebgcn_lrp_class0_td_conv2_edge)
            ebgcn_lrp_class1_td_edge, e1 = get_edge_ranked_list(ebgcn_lrp_class1_td_conv1_edge,
                                                                ebgcn_lrp_class1_td_conv2_edge)
            ebgcn_lrp_class2_td_edge, e2 = get_edge_ranked_list(ebgcn_lrp_class1_td_conv1_edge,
                                                                ebgcn_lrp_class2_td_conv2_edge)
            ebgcn_lrp_class3_td_edge, e3 = get_edge_ranked_list(ebgcn_lrp_class1_td_conv1_edge,
                                                                ebgcn_lrp_class3_td_conv2_edge)
            ebgcn_lrp_allclass_td_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                         np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
            ebgcn_lrp_allclass_td_edge /= ebgcn_lrp_allclass_td_edge.sum()
            ebgcn_lrp_allclass_td_edge = torch.topk(torch.as_tensor(ebgcn_lrp_allclass_td_edge),
                                                    k=ebgcn_lrp_allclass_td_edge.shape[0])
            ebgcn_lrp_allclass_td_edge = [ebgcn_lrp_allclass_td_edge.indices.tolist(),
                                          ebgcn_lrp_allclass_td_edge.values.tolist()]
            # BU
            ebgcn_bu_conv1_text = ebgcn_explain_json['bu_gcn_explanations']['conv1_text'][0]
            ebgcn_bu_conv2_text = ebgcn_explain_json['bu_gcn_explanations']['conv2_text'][0]
            ebgcn_bu_conv1_edge_weights = ebgcn_explain_json['bu_gcn_explanations']['conv1_edge_weights']
            ebgcn_bu_conv2_edge_weights = ebgcn_explain_json['bu_gcn_explanations']['conv2_edge_weights']

            ebgcn_lrp_class0_bu_conv1 = ebgcn_explain_json['lrp_class0_bu_conv1']
            ebgcn_lrp_class1_bu_conv1 = ebgcn_explain_json['lrp_class1_bu_conv1']
            ebgcn_lrp_class2_bu_conv1 = ebgcn_explain_json['lrp_class2_bu_conv1']
            ebgcn_lrp_class3_bu_conv1 = ebgcn_explain_json['lrp_class3_bu_conv1']

            ebgcn_lrp_class0_bu_conv2 = ebgcn_explain_json['lrp_class0_bu_conv2']
            ebgcn_lrp_class1_bu_conv2 = ebgcn_explain_json['lrp_class1_bu_conv2']
            ebgcn_lrp_class2_bu_conv2 = ebgcn_explain_json['lrp_class2_bu_conv2']
            ebgcn_lrp_class3_bu_conv2 = ebgcn_explain_json['lrp_class3_bu_conv2']

            logits = ebgcn_explain_json['logits']
            logit_weights = np.exp(np.asarray(logits))[0]

            # Conv 1
            ebgcn_lrp_class0_bu_conv1_text, ebgcn_lrp_class0_bu_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv1_text,
                *ebgcn_bu_conv1_edge_weights, ebgcn_lrp_class0_bu_conv1)
            ebgcn_lrp_class1_bu_conv1_text, ebgcn_lrp_class1_bu_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv1_text,
                *ebgcn_bu_conv1_edge_weights, ebgcn_lrp_class1_bu_conv1)
            ebgcn_lrp_class2_bu_conv1_text, ebgcn_lrp_class2_bu_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv1_text,
                *ebgcn_bu_conv1_edge_weights, ebgcn_lrp_class2_bu_conv1)
            ebgcn_lrp_class3_bu_conv1_text, ebgcn_lrp_class3_bu_conv1_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv1_text,
                *ebgcn_bu_conv1_edge_weights, ebgcn_lrp_class3_bu_conv1)

            # Conv 2
            ebgcn_lrp_class0_bu_conv2_text, ebgcn_lrp_class0_bu_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv2_text,
                *ebgcn_bu_conv2_edge_weights, ebgcn_lrp_class0_bu_conv2)
            ebgcn_lrp_class1_bu_conv2_text, ebgcn_lrp_class1_bu_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv2_text,
                *ebgcn_bu_conv2_edge_weights, ebgcn_lrp_class1_bu_conv2)
            ebgcn_lrp_class2_bu_conv2_text, ebgcn_lrp_class2_bu_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv2_text,
                *ebgcn_bu_conv2_edge_weights, ebgcn_lrp_class2_bu_conv2)
            ebgcn_lrp_class3_bu_conv2_text, ebgcn_lrp_class3_bu_conv2_edge, _ = compute_gcl_relevance(
                ebgcn_bu_conv2_text,
                *ebgcn_bu_conv2_edge_weights, ebgcn_lrp_class3_bu_conv2)

            ebgcn_lrp_class0_bu_text, t0 = get_text_ranked_list(ebgcn_lrp_class0_bu_conv1_text,
                                                                ebgcn_lrp_class0_bu_conv2_text)
            ebgcn_lrp_class1_bu_text, t1 = get_text_ranked_list(ebgcn_lrp_class1_bu_conv1_text,
                                                                ebgcn_lrp_class1_bu_conv2_text)
            ebgcn_lrp_class2_bu_text, t2 = get_text_ranked_list(ebgcn_lrp_class1_bu_conv1_text,
                                                                ebgcn_lrp_class2_bu_conv2_text)
            ebgcn_lrp_class3_bu_text, t3 = get_text_ranked_list(ebgcn_lrp_class1_bu_conv1_text,
                                                                ebgcn_lrp_class3_bu_conv2_text)
            ebgcn_lrp_allclass_bu_text = np.asarray(t0) * logit_weights[0] + np.asarray(t1) * logit_weights[1] + \
                                         np.asarray(t2) * logit_weights[2] + np.asarray(t3) * logit_weights[3]
            ebgcn_lrp_allclass_bu_text /= ebgcn_lrp_allclass_bu_text.sum()
            ebgcn_lrp_allclass_bu_text = torch.topk(torch.as_tensor(ebgcn_lrp_allclass_bu_text),
                                                    k=ebgcn_lrp_allclass_bu_text.shape[0])
            ebgcn_lrp_allclass_bu_text = [ebgcn_lrp_allclass_bu_text.indices.tolist(),
                                          ebgcn_lrp_allclass_bu_text.values.tolist()]

            ebgcn_lrp_class0_bu_edge, e0 = get_edge_ranked_list(ebgcn_lrp_class0_bu_conv1_edge,
                                                                ebgcn_lrp_class0_bu_conv2_edge)
            ebgcn_lrp_class1_bu_edge, e1 = get_edge_ranked_list(ebgcn_lrp_class1_bu_conv1_edge,
                                                                ebgcn_lrp_class1_bu_conv2_edge)
            ebgcn_lrp_class2_bu_edge, e2 = get_edge_ranked_list(ebgcn_lrp_class1_bu_conv1_edge,
                                                                ebgcn_lrp_class2_bu_conv2_edge)
            ebgcn_lrp_class3_bu_edge, e3 = get_edge_ranked_list(ebgcn_lrp_class1_bu_conv1_edge,
                                                                ebgcn_lrp_class3_bu_conv2_edge)
            ebgcn_lrp_allclass_bu_edge = np.asarray(e0) * logit_weights[0] + np.asarray(e1) * logit_weights[1] + \
                                         np.asarray(e2) * logit_weights[2] + np.asarray(e3) * logit_weights[3]
            ebgcn_lrp_allclass_bu_edge /= ebgcn_lrp_allclass_bu_edge.sum()
            ebgcn_lrp_allclass_bu_edge = torch.topk(torch.as_tensor(ebgcn_lrp_allclass_bu_edge),
                                                    k=ebgcn_lrp_allclass_bu_edge.shape[0])
            ebgcn_lrp_allclass_bu_edge = [ebgcn_lrp_allclass_bu_edge.indices.tolist(),
                                          ebgcn_lrp_allclass_bu_edge.values.tolist()]

            # Whole Model
            ebgcn_lrp_class0 = ebgcn_explain_json['lrp_class0_top_k']
            ebgcn_lrp_class1 = ebgcn_explain_json['lrp_class1_top_k']
            ebgcn_lrp_class2 = ebgcn_explain_json['lrp_class2_top_k']
            ebgcn_lrp_class3 = ebgcn_explain_json['lrp_class3_top_k']
            ebgcn_lrp_allclass = ebgcn_explain_json['lrp_allclass_top_k']

            # Extra
            ebgcn_nr_fc = np.asarray(ebgcn_explain_json['lrp_class0_fc'])
            ebgcn_fr_fc = np.asarray(ebgcn_explain_json['lrp_class1_fc'])
            ebgcn_tr_fc = np.asarray(ebgcn_explain_json['lrp_class2_fc'])
            ebgcn_ur_fc = np.asarray(ebgcn_explain_json['lrp_class3_fc'])
            ebgcn_nr_fc = [ebgcn_nr_fc[:ebgcn_nr_fc.shape[-1] // 2].sum(),
                           ebgcn_nr_fc[ebgcn_nr_fc.shape[-1] // 2:].sum()]
            ebgcn_fr_fc = [ebgcn_fr_fc[:ebgcn_fr_fc.shape[-1] // 2].sum(),
                           ebgcn_fr_fc[ebgcn_fr_fc.shape[-1] // 2:].sum()]
            ebgcn_tr_fc = [ebgcn_tr_fc[:ebgcn_tr_fc.shape[-1] // 2].sum(),
                           ebgcn_tr_fc[ebgcn_tr_fc.shape[-1] // 2:].sum()]
            ebgcn_ur_fc = [ebgcn_ur_fc[:ebgcn_ur_fc.shape[-1] // 2].sum(),
                           ebgcn_ur_fc[ebgcn_ur_fc.shape[-1] // 2:].sum()]

            ebgcn_nr_td_fc = np.asarray(ebgcn_explain_json['lrp_class0_td_conv1_fc'])
            ebgcn_fr_td_fc = np.asarray(ebgcn_explain_json['lrp_class1_td_conv1_fc'])
            ebgcn_tr_td_fc = np.asarray(ebgcn_explain_json['lrp_class2_td_conv1_fc'])
            ebgcn_ur_td_fc = np.asarray(ebgcn_explain_json['lrp_class3_td_conv1_fc'])
            ebgcn_nr_td_fc = [list(range(ebgcn_nr_td_fc.shape[0])), ebgcn_nr_td_fc.sum(-1)]
            ebgcn_fr_td_fc = [list(range(ebgcn_fr_td_fc.shape[0])), ebgcn_fr_td_fc.sum(-1)]
            ebgcn_tr_td_fc = [list(range(ebgcn_tr_td_fc.shape[0])), ebgcn_tr_td_fc.sum(-1)]
            ebgcn_ur_td_fc = [list(range(ebgcn_ur_td_fc.shape[0])), ebgcn_ur_td_fc.sum(-1)]

            ebgcn_nr_bu_fc = np.asarray(ebgcn_explain_json['lrp_class0_bu_conv1_fc'])
            ebgcn_fr_bu_fc = np.asarray(ebgcn_explain_json['lrp_class1_bu_conv1_fc'])
            ebgcn_tr_bu_fc = np.asarray(ebgcn_explain_json['lrp_class2_bu_conv1_fc'])
            ebgcn_ur_bu_fc = np.asarray(ebgcn_explain_json['lrp_class3_bu_conv1_fc'])
            ebgcn_nr_bu_fc = [list(range(ebgcn_nr_bu_fc.shape[0])), ebgcn_nr_bu_fc.sum(-1)]
            ebgcn_fr_bu_fc = [list(range(ebgcn_fr_bu_fc.shape[0])), ebgcn_fr_bu_fc.sum(-1)]
            ebgcn_tr_bu_fc = [list(range(ebgcn_tr_bu_fc.shape[0])), ebgcn_tr_bu_fc.sum(-1)]
            ebgcn_ur_bu_fc = [list(range(ebgcn_ur_bu_fc.shape[0])), ebgcn_ur_bu_fc.sum(-1)]

            # Get sublist
            ebgcn_lrp_class0_td_text = get_sublist(*ebgcn_lrp_class0_td_text, select, k, quantile, interval)
            ebgcn_lrp_class1_td_text = get_sublist(*ebgcn_lrp_class1_td_text, select, k, quantile, interval)
            ebgcn_lrp_class2_td_text = get_sublist(*ebgcn_lrp_class2_td_text, select, k, quantile, interval)
            ebgcn_lrp_class3_td_text = get_sublist(*ebgcn_lrp_class3_td_text, select, k, quantile, interval)
            ebgcn_lrp_allclass_td_text = get_sublist(*ebgcn_lrp_allclass_td_text, select, k, quantile, interval)

            ebgcn_lrp_class0_td_edge = get_sublist(*ebgcn_lrp_class0_td_edge, select, k, quantile, interval)
            ebgcn_lrp_class1_td_edge = get_sublist(*ebgcn_lrp_class1_td_edge, select, k, quantile, interval)
            ebgcn_lrp_class2_td_edge = get_sublist(*ebgcn_lrp_class2_td_edge, select, k, quantile, interval)
            ebgcn_lrp_class3_td_edge = get_sublist(*ebgcn_lrp_class3_td_edge, select, k, quantile, interval)
            ebgcn_lrp_allclass_td_edge = get_sublist(*ebgcn_lrp_allclass_td_edge, select, k, quantile, interval)

            ebgcn_lrp_class0_bu_text = get_sublist(*ebgcn_lrp_class0_bu_text, select, k, quantile, interval)
            ebgcn_lrp_class1_bu_text = get_sublist(*ebgcn_lrp_class1_bu_text, select, k, quantile, interval)
            ebgcn_lrp_class2_bu_text = get_sublist(*ebgcn_lrp_class2_bu_text, select, k, quantile, interval)
            ebgcn_lrp_class3_bu_text = get_sublist(*ebgcn_lrp_class3_bu_text, select, k, quantile, interval)
            ebgcn_lrp_allclass_bu_text = get_sublist(*ebgcn_lrp_allclass_bu_text, select, k, quantile, interval)

            ebgcn_lrp_class0_bu_edge = get_sublist(*ebgcn_lrp_class0_bu_edge, select, k, quantile, interval)
            ebgcn_lrp_class1_bu_edge = get_sublist(*ebgcn_lrp_class1_bu_edge, select, k, quantile, interval)
            ebgcn_lrp_class2_bu_edge = get_sublist(*ebgcn_lrp_class2_bu_edge, select, k, quantile, interval)
            ebgcn_lrp_class3_bu_edge = get_sublist(*ebgcn_lrp_class3_bu_edge, select, k, quantile, interval)
            ebgcn_lrp_allclass_bu_edge = get_sublist(*ebgcn_lrp_allclass_bu_edge, select, k, quantile, interval)

            ebgcn_lrp_class0 = get_sublist(*ebgcn_lrp_class0, select, k, quantile, interval)
            ebgcn_lrp_class1 = get_sublist(*ebgcn_lrp_class1, select, k, quantile, interval)
            ebgcn_lrp_class2 = get_sublist(*ebgcn_lrp_class2, select, k, quantile, interval)
            ebgcn_lrp_class3 = get_sublist(*ebgcn_lrp_class3, select, k, quantile, interval)
            ebgcn_lrp_allclass = get_sublist(*ebgcn_lrp_allclass, select, k, quantile, interval)

            ebgcn_nr_td_fc = get_sublist(*ebgcn_nr_td_fc, select, k, quantile, interval)
            ebgcn_fr_td_fc = get_sublist(*ebgcn_fr_td_fc, select, k, quantile, interval)
            ebgcn_tr_td_fc = get_sublist(*ebgcn_tr_td_fc, select, k, quantile, interval)
            ebgcn_ur_td_fc = get_sublist(*ebgcn_ur_td_fc, select, k, quantile, interval)

            ebgcn_nr_bu_fc = get_sublist(*ebgcn_nr_bu_fc, select, k, quantile, interval)
            ebgcn_fr_bu_fc = get_sublist(*ebgcn_fr_bu_fc, select, k, quantile, interval)
            ebgcn_tr_bu_fc = get_sublist(*ebgcn_tr_bu_fc, select, k, quantile, interval)
            ebgcn_ur_bu_fc = get_sublist(*ebgcn_ur_bu_fc, select, k, quantile, interval)

            vec = [bigcn_lrp_class0_td_text, bigcn_lrp_class1_td_text, bigcn_lrp_class2_td_text,
                   bigcn_lrp_class3_td_text,
                   bigcn_lrp_allclass_td_text,
                   ebgcn_lrp_class0_td_text, ebgcn_lrp_class1_td_text, ebgcn_lrp_class2_td_text,
                   ebgcn_lrp_class3_td_text,
                   ebgcn_lrp_allclass_td_text,
                   bigcn_lrp_class0_bu_text, bigcn_lrp_class1_bu_text, bigcn_lrp_class2_bu_text,
                   bigcn_lrp_class3_bu_text,
                   bigcn_lrp_allclass_bu_text,
                   ebgcn_lrp_class0_bu_text, ebgcn_lrp_class1_bu_text, ebgcn_lrp_class2_bu_text,
                   ebgcn_lrp_class3_bu_text,
                   ebgcn_lrp_allclass_bu_text,
                   bigcn_lrp_class0_td_edge, bigcn_lrp_class1_td_edge, bigcn_lrp_class2_td_edge,
                   bigcn_lrp_class3_td_edge,
                   bigcn_lrp_allclass_td_edge,
                   ebgcn_lrp_class0_td_edge, ebgcn_lrp_class1_td_edge, ebgcn_lrp_class2_td_edge,
                   ebgcn_lrp_class3_td_edge,
                   ebgcn_lrp_allclass_td_edge,
                   bigcn_lrp_class0_bu_edge, bigcn_lrp_class1_bu_edge, bigcn_lrp_class2_bu_edge,
                   bigcn_lrp_class3_bu_edge,
                   bigcn_lrp_allclass_bu_edge,
                   ebgcn_lrp_class0_bu_edge, ebgcn_lrp_class1_bu_edge, ebgcn_lrp_class2_bu_edge,
                   ebgcn_lrp_class3_bu_edge,
                   ebgcn_lrp_allclass_bu_edge,
                   bigcn_lrp_class0, bigcn_lrp_class1, bigcn_lrp_class2, bigcn_lrp_class3, bigcn_lrp_allclass,
                   ebgcn_lrp_class0, ebgcn_lrp_class1, ebgcn_lrp_class2, ebgcn_lrp_class3, ebgcn_lrp_allclass]
            vec2 = [bigcn_nr_td_fc, bigcn_fr_td_fc, bigcn_tr_td_fc, bigcn_ur_td_fc,
                    bigcn_nr_bu_fc, bigcn_fr_bu_fc, bigcn_tr_bu_fc, bigcn_ur_bu_fc,
                    ebgcn_nr_td_fc, ebgcn_fr_td_fc, ebgcn_tr_td_fc, ebgcn_ur_td_fc,
                    ebgcn_nr_bu_fc, ebgcn_fr_bu_fc, ebgcn_tr_bu_fc, ebgcn_ur_bu_fc]
            vec3 = [bigcn_nr_fc, bigcn_fr_fc, bigcn_tr_fc, bigcn_ur_fc,
                    ebgcn_nr_fc, ebgcn_fr_fc, ebgcn_tr_fc, ebgcn_ur_fc]
            return vec + vec2, vec3

        def load_model_explanations_twitter1516(bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log):
            try:
                # BiGCN
                with open(bigcn_explain_json_path, 'r') as f:
                    bigcn_explain_json = json.load(f)
                if len(bigcn_explain_json[f'{exp_method}_allclass_top_k'][0]) < min_graph_size:
                    error_log.append(f'Event {tree_id} graph smaller than {min_graph_size}')
                    return None, None
            except FileNotFoundError:
                error_log.append(f'Error: Missing BiGCN explanation files for Tree num {tree_id}\t'
                                 f'{bigcn_explain_json_path}')
                return None, None
            except IOError:
                error_log.append(f'Error: Unable to load BiGCN explanation files for Tree num {tree_id}\t'
                                 f'{bigcn_explain_json_path}')
                return None, None
            except KeyError:
                try:
                    print(list(bigcn_explain_json.keys))
                except Exception as err:
                    print(err, traceback.print_exc())
                    print(bigcn_explain_json)
                raise Exception
                return None, None
            except Exception as err:
                print(err, traceback.print_exc())
                error_log.append(f'Error: Unknown error while attempting to load BiGCN explanation files for Tree num '
                                 f'{tree_id}\t{bigcn_explain_json_path}')
                return None, None
            try:
                bigcn_lrp_class0 = bigcn_explain_json[f'{exp_method}_class0_top_k']
                bigcn_lrp_class1 = bigcn_explain_json[f'{exp_method}_class1_top_k']
                bigcn_lrp_class2 = bigcn_explain_json[f'{exp_method}_class2_top_k']
                bigcn_lrp_class3 = bigcn_explain_json[f'{exp_method}_class3_top_k']
                bigcn_lrp_allclass = bigcn_explain_json[f'{exp_method}_allclass_top_k']

                # Extra
                bigcn_nr_fc = bigcn_explain_json['nr_fc']
                bigcn_fr_fc = bigcn_explain_json['fr_fc']
                bigcn_tr_fc = bigcn_explain_json['tr_fc']
                bigcn_ur_fc = bigcn_explain_json['ur_fc']

                bigcn_nr_td_fc = bigcn_explain_json['nr_td_fc']
                bigcn_fr_td_fc = bigcn_explain_json['fr_td_fc']
                bigcn_tr_td_fc = bigcn_explain_json['tr_td_fc']
                bigcn_ur_td_fc = bigcn_explain_json['ur_td_fc']

                bigcn_nr_bu_fc = bigcn_explain_json['nr_bu_fc']
                bigcn_fr_bu_fc = bigcn_explain_json['fr_bu_fc']
                bigcn_tr_bu_fc = bigcn_explain_json['tr_bu_fc']
                bigcn_ur_bu_fc = bigcn_explain_json['ur_bu_fc']

                # TD Text
                bigcn_lrp_class0_td_text = bigcn_explain_json[f'{exp_method}_class0_td_text']
                bigcn_lrp_class1_td_text = bigcn_explain_json[f'{exp_method}_class1_td_text']
                bigcn_lrp_class2_td_text = bigcn_explain_json[f'{exp_method}_class2_td_text']
                bigcn_lrp_class3_td_text = bigcn_explain_json[f'{exp_method}_class3_td_text']
                bigcn_lrp_allclass_td_text = bigcn_explain_json[f'{exp_method}_allclass_td_text']
                # TD Edge
                bigcn_lrp_class0_td_edge = bigcn_explain_json[f'{exp_method}_class0_td_edge']
                bigcn_lrp_class1_td_edge = bigcn_explain_json[f'{exp_method}_class1_td_edge']
                bigcn_lrp_class2_td_edge = bigcn_explain_json[f'{exp_method}_class2_td_edge']
                bigcn_lrp_class3_td_edge = bigcn_explain_json[f'{exp_method}_class3_td_edge']
                bigcn_lrp_allclass_td_edge = bigcn_explain_json[f'{exp_method}_allclass_td_edge']

                # BU Text
                bigcn_lrp_class0_bu_text = bigcn_explain_json[f'{exp_method}_class0_bu_text']
                bigcn_lrp_class1_bu_text = bigcn_explain_json[f'{exp_method}_class1_bu_text']
                bigcn_lrp_class2_bu_text = bigcn_explain_json[f'{exp_method}_class2_bu_text']
                bigcn_lrp_class3_bu_text = bigcn_explain_json[f'{exp_method}_class3_bu_text']
                bigcn_lrp_allclass_bu_text = bigcn_explain_json[f'{exp_method}_allclass_bu_text']
                # BU Edge
                bigcn_lrp_class0_bu_edge = bigcn_explain_json[f'{exp_method}_class0_bu_edge']
                bigcn_lrp_class1_bu_edge = bigcn_explain_json[f'{exp_method}_class1_bu_edge']
                bigcn_lrp_class2_bu_edge = bigcn_explain_json[f'{exp_method}_class2_bu_edge']
                bigcn_lrp_class3_bu_edge = bigcn_explain_json[f'{exp_method}_class3_bu_edge']
                bigcn_lrp_allclass_bu_edge = bigcn_explain_json[f'{exp_method}_allclass_bu_edge']

                # Get sublist
                bigcn_lrp_class0_td_text = get_sublist(*bigcn_lrp_class0_td_text, select, k, quantile, interval)
                bigcn_lrp_class1_td_text = get_sublist(*bigcn_lrp_class1_td_text, select, k, quantile, interval)
                bigcn_lrp_class2_td_text = get_sublist(*bigcn_lrp_class2_td_text, select, k, quantile, interval)
                bigcn_lrp_class3_td_text = get_sublist(*bigcn_lrp_class3_td_text, select, k, quantile, interval)
                bigcn_lrp_allclass_td_text = get_sublist(*bigcn_lrp_allclass_td_text, select, k, quantile, interval)

                bigcn_lrp_class0_td_edge = get_sublist(*bigcn_lrp_class0_td_edge, select, k, quantile, interval)
                bigcn_lrp_class1_td_edge = get_sublist(*bigcn_lrp_class1_td_edge, select, k, quantile, interval)
                bigcn_lrp_class2_td_edge = get_sublist(*bigcn_lrp_class2_td_edge, select, k, quantile, interval)
                bigcn_lrp_class3_td_edge = get_sublist(*bigcn_lrp_class3_td_edge, select, k, quantile, interval)
                bigcn_lrp_allclass_td_edge = get_sublist(*bigcn_lrp_allclass_td_edge, select, k, quantile, interval)

                bigcn_lrp_class0_bu_text = get_sublist(*bigcn_lrp_class0_bu_text, select, k, quantile, interval)
                bigcn_lrp_class1_bu_text = get_sublist(*bigcn_lrp_class1_bu_text, select, k, quantile, interval)
                bigcn_lrp_class2_bu_text = get_sublist(*bigcn_lrp_class2_bu_text, select, k, quantile, interval)
                bigcn_lrp_class3_bu_text = get_sublist(*bigcn_lrp_class3_bu_text, select, k, quantile, interval)
                bigcn_lrp_allclass_bu_text = get_sublist(*bigcn_lrp_allclass_bu_text, select, k, quantile, interval)

                bigcn_lrp_class0_bu_edge = get_sublist(*bigcn_lrp_class0_bu_edge, select, k, quantile, interval)
                bigcn_lrp_class1_bu_edge = get_sublist(*bigcn_lrp_class1_bu_edge, select, k, quantile, interval)
                bigcn_lrp_class2_bu_edge = get_sublist(*bigcn_lrp_class2_bu_edge, select, k, quantile, interval)
                bigcn_lrp_class3_bu_edge = get_sublist(*bigcn_lrp_class3_bu_edge, select, k, quantile, interval)
                bigcn_lrp_allclass_bu_edge = get_sublist(*bigcn_lrp_allclass_bu_edge, select, k, quantile, interval)

                bigcn_lrp_class0 = get_sublist(*bigcn_lrp_class0, select, k, quantile, interval)
                bigcn_lrp_class1 = get_sublist(*bigcn_lrp_class1, select, k, quantile, interval)
                bigcn_lrp_class2 = get_sublist(*bigcn_lrp_class2, select, k, quantile, interval)
                bigcn_lrp_class3 = get_sublist(*bigcn_lrp_class3, select, k, quantile, interval)
                bigcn_lrp_allclass = get_sublist(*bigcn_lrp_allclass, select, k, quantile, interval)

                bigcn_nr_td_fc = get_sublist(*bigcn_nr_td_fc, select, k, quantile, interval)
                bigcn_fr_td_fc = get_sublist(*bigcn_fr_td_fc, select, k, quantile, interval)
                bigcn_tr_td_fc = get_sublist(*bigcn_tr_td_fc, select, k, quantile, interval)
                bigcn_ur_td_fc = get_sublist(*bigcn_ur_td_fc, select, k, quantile, interval)

                bigcn_nr_bu_fc = get_sublist(*bigcn_nr_bu_fc, select, k, quantile, interval)
                bigcn_fr_bu_fc = get_sublist(*bigcn_fr_bu_fc, select, k, quantile, interval)
                bigcn_tr_bu_fc = get_sublist(*bigcn_tr_bu_fc, select, k, quantile, interval)
                bigcn_ur_bu_fc = get_sublist(*bigcn_ur_bu_fc, select, k, quantile, interval)
            except Exception as e:
                print(e, traceback.print_exc())
                traceback.print_stack()
                raise Exception
            try:
                # EBGCN
                with open(ebgcn_explain_json_path, 'r') as f:
                    ebgcn_explain_json = json.load(f)
                if len(ebgcn_explain_json[f'{exp_method}_allclass_top_k'][0]) < min_graph_size:
                    error_log.append(f'Event {tree_id} graph smaller than {min_graph_size}')
                    return None, None
            except FileNotFoundError:
                error_log.append(f'Error: Missing EBGCN explanation files for Tree num {tree_id}\t'
                                 f'{ebgcn_explain_json_path}')
                return None, None
            except IOError:
                error_log.append(f'Error: Unable to load EBGCN explanation files for Tree num {tree_id}\t'
                                 f'{ebgcn_explain_json_path}')
                return None, None
            except KeyError:
                try:
                    print(list(ebgcn_explain_json.keys))
                except Exception as err:
                    print(err, traceback.print_exc())
                    print(ebgcn_explain_json)
                raise Exception
                return None, None
            except Exception as err:
                print(err, traceback.print_exc())
                error_log.append(f'Error: Unknown error while attempting to load EBGCN explanation files for Tree num '
                                 f'{tree_id}\t{ebgcn_explain_json_path}')
                return None, None
            try:
                ebgcn_lrp_class0 = ebgcn_explain_json[f'{exp_method}_class0_top_k']
                ebgcn_lrp_class1 = ebgcn_explain_json[f'{exp_method}_class1_top_k']
                ebgcn_lrp_class2 = ebgcn_explain_json[f'{exp_method}_class2_top_k']
                ebgcn_lrp_class3 = ebgcn_explain_json[f'{exp_method}_class3_top_k']
                ebgcn_lrp_allclass = ebgcn_explain_json[f'{exp_method}_allclass_top_k']

                # Extra
                ebgcn_nr_fc = ebgcn_explain_json['nr_fc']
                ebgcn_fr_fc = ebgcn_explain_json['fr_fc']
                ebgcn_tr_fc = ebgcn_explain_json['tr_fc']
                ebgcn_ur_fc = ebgcn_explain_json['ur_fc']

                ebgcn_nr_td_fc = ebgcn_explain_json['nr_td_fc']
                ebgcn_fr_td_fc = ebgcn_explain_json['fr_td_fc']
                ebgcn_tr_td_fc = ebgcn_explain_json['tr_td_fc']
                ebgcn_ur_td_fc = ebgcn_explain_json['ur_td_fc']

                ebgcn_nr_bu_fc = ebgcn_explain_json['nr_bu_fc']
                ebgcn_fr_bu_fc = ebgcn_explain_json['fr_bu_fc']
                ebgcn_tr_bu_fc = ebgcn_explain_json['tr_bu_fc']
                ebgcn_ur_bu_fc = ebgcn_explain_json['ur_bu_fc']

                # TD Text
                ebgcn_lrp_class0_td_text = ebgcn_explain_json[f'{exp_method}_class0_td_text']
                ebgcn_lrp_class1_td_text = ebgcn_explain_json[f'{exp_method}_class1_td_text']
                ebgcn_lrp_class2_td_text = ebgcn_explain_json[f'{exp_method}_class2_td_text']
                ebgcn_lrp_class3_td_text = ebgcn_explain_json[f'{exp_method}_class3_td_text']
                ebgcn_lrp_allclass_td_text = ebgcn_explain_json[f'{exp_method}_allclass_td_text']
                # TD Edge
                ebgcn_lrp_class0_td_edge = ebgcn_explain_json[f'{exp_method}_class0_td_edge']
                ebgcn_lrp_class1_td_edge = ebgcn_explain_json[f'{exp_method}_class1_td_edge']
                ebgcn_lrp_class2_td_edge = ebgcn_explain_json[f'{exp_method}_class2_td_edge']
                ebgcn_lrp_class3_td_edge = ebgcn_explain_json[f'{exp_method}_class3_td_edge']
                ebgcn_lrp_allclass_td_edge = ebgcn_explain_json[f'{exp_method}_allclass_td_edge']

                # BU Text
                ebgcn_lrp_class0_bu_text = ebgcn_explain_json[f'{exp_method}_class0_bu_text']
                ebgcn_lrp_class1_bu_text = ebgcn_explain_json[f'{exp_method}_class1_bu_text']
                ebgcn_lrp_class2_bu_text = ebgcn_explain_json[f'{exp_method}_class2_bu_text']
                ebgcn_lrp_class3_bu_text = ebgcn_explain_json[f'{exp_method}_class3_bu_text']
                ebgcn_lrp_allclass_bu_text = ebgcn_explain_json[f'{exp_method}_allclass_bu_text']
                # BU Edge
                ebgcn_lrp_class0_bu_edge = ebgcn_explain_json[f'{exp_method}_class0_bu_edge']
                ebgcn_lrp_class1_bu_edge = ebgcn_explain_json[f'{exp_method}_class1_bu_edge']
                ebgcn_lrp_class2_bu_edge = ebgcn_explain_json[f'{exp_method}_class2_bu_edge']
                ebgcn_lrp_class3_bu_edge = ebgcn_explain_json[f'{exp_method}_class3_bu_edge']
                ebgcn_lrp_allclass_bu_edge = ebgcn_explain_json[f'{exp_method}_allclass_bu_edge']

                # Get sublist
                ebgcn_lrp_class0_td_text = get_sublist(*ebgcn_lrp_class0_td_text, select, k, quantile, interval)
                ebgcn_lrp_class1_td_text = get_sublist(*ebgcn_lrp_class1_td_text, select, k, quantile, interval)
                ebgcn_lrp_class2_td_text = get_sublist(*ebgcn_lrp_class2_td_text, select, k, quantile, interval)
                ebgcn_lrp_class3_td_text = get_sublist(*ebgcn_lrp_class3_td_text, select, k, quantile, interval)
                ebgcn_lrp_allclass_td_text = get_sublist(*ebgcn_lrp_allclass_td_text, select, k, quantile, interval)

                ebgcn_lrp_class0_td_edge = get_sublist(*ebgcn_lrp_class0_td_edge, select, k, quantile, interval)
                ebgcn_lrp_class1_td_edge = get_sublist(*ebgcn_lrp_class1_td_edge, select, k, quantile, interval)
                ebgcn_lrp_class2_td_edge = get_sublist(*ebgcn_lrp_class2_td_edge, select, k, quantile, interval)
                ebgcn_lrp_class3_td_edge = get_sublist(*ebgcn_lrp_class3_td_edge, select, k, quantile, interval)
                ebgcn_lrp_allclass_td_edge = get_sublist(*ebgcn_lrp_allclass_td_edge, select, k, quantile, interval)

                ebgcn_lrp_class0_bu_text = get_sublist(*ebgcn_lrp_class0_bu_text, select, k, quantile, interval)
                ebgcn_lrp_class1_bu_text = get_sublist(*ebgcn_lrp_class1_bu_text, select, k, quantile, interval)
                ebgcn_lrp_class2_bu_text = get_sublist(*ebgcn_lrp_class2_bu_text, select, k, quantile, interval)
                ebgcn_lrp_class3_bu_text = get_sublist(*ebgcn_lrp_class3_bu_text, select, k, quantile, interval)
                ebgcn_lrp_allclass_bu_text = get_sublist(*ebgcn_lrp_allclass_bu_text, select, k, quantile, interval)

                ebgcn_lrp_class0_bu_edge = get_sublist(*ebgcn_lrp_class0_bu_edge, select, k, quantile, interval)
                ebgcn_lrp_class1_bu_edge = get_sublist(*ebgcn_lrp_class1_bu_edge, select, k, quantile, interval)
                ebgcn_lrp_class2_bu_edge = get_sublist(*ebgcn_lrp_class2_bu_edge, select, k, quantile, interval)
                ebgcn_lrp_class3_bu_edge = get_sublist(*ebgcn_lrp_class3_bu_edge, select, k, quantile, interval)
                ebgcn_lrp_allclass_bu_edge = get_sublist(*ebgcn_lrp_allclass_bu_edge, select, k, quantile, interval)

                ebgcn_lrp_class0 = get_sublist(*ebgcn_lrp_class0, select, k, quantile, interval)
                ebgcn_lrp_class1 = get_sublist(*ebgcn_lrp_class1, select, k, quantile, interval)
                ebgcn_lrp_class2 = get_sublist(*ebgcn_lrp_class2, select, k, quantile, interval)
                ebgcn_lrp_class3 = get_sublist(*ebgcn_lrp_class3, select, k, quantile, interval)
                ebgcn_lrp_allclass = get_sublist(*ebgcn_lrp_allclass, select, k, quantile, interval)

                ebgcn_nr_td_fc = get_sublist(*ebgcn_nr_td_fc, select, k, quantile, interval)
                ebgcn_fr_td_fc = get_sublist(*ebgcn_fr_td_fc, select, k, quantile, interval)
                ebgcn_tr_td_fc = get_sublist(*ebgcn_tr_td_fc, select, k, quantile, interval)
                ebgcn_ur_td_fc = get_sublist(*ebgcn_ur_td_fc, select, k, quantile, interval)

                ebgcn_nr_bu_fc = get_sublist(*ebgcn_nr_bu_fc, select, k, quantile, interval)
                ebgcn_fr_bu_fc = get_sublist(*ebgcn_fr_bu_fc, select, k, quantile, interval)
                ebgcn_tr_bu_fc = get_sublist(*ebgcn_tr_bu_fc, select, k, quantile, interval)
                ebgcn_ur_bu_fc = get_sublist(*ebgcn_ur_bu_fc, select, k, quantile, interval)
            except Exception as e:
                print(e, traceback.print_exc())
                traceback.print_stack()
                raise Exception
            vec = [bigcn_lrp_class0_td_text, bigcn_lrp_class1_td_text, bigcn_lrp_class2_td_text,
                   bigcn_lrp_class3_td_text,
                   bigcn_lrp_allclass_td_text,
                   ebgcn_lrp_class0_td_text, ebgcn_lrp_class1_td_text, ebgcn_lrp_class2_td_text,
                   ebgcn_lrp_class3_td_text,
                   ebgcn_lrp_allclass_td_text,
                   bigcn_lrp_class0_bu_text, bigcn_lrp_class1_bu_text, bigcn_lrp_class2_bu_text,
                   bigcn_lrp_class3_bu_text,
                   bigcn_lrp_allclass_bu_text,
                   ebgcn_lrp_class0_bu_text, ebgcn_lrp_class1_bu_text, ebgcn_lrp_class2_bu_text,
                   ebgcn_lrp_class3_bu_text,
                   ebgcn_lrp_allclass_bu_text,
                   bigcn_lrp_class0_td_edge, bigcn_lrp_class1_td_edge, bigcn_lrp_class2_td_edge,
                   bigcn_lrp_class3_td_edge,
                   bigcn_lrp_allclass_td_edge,
                   ebgcn_lrp_class0_td_edge, ebgcn_lrp_class1_td_edge, ebgcn_lrp_class2_td_edge,
                   ebgcn_lrp_class3_td_edge,
                   ebgcn_lrp_allclass_td_edge,
                   bigcn_lrp_class0_bu_edge, bigcn_lrp_class1_bu_edge, bigcn_lrp_class2_bu_edge,
                   bigcn_lrp_class3_bu_edge,
                   bigcn_lrp_allclass_bu_edge,
                   ebgcn_lrp_class0_bu_edge, ebgcn_lrp_class1_bu_edge, ebgcn_lrp_class2_bu_edge,
                   ebgcn_lrp_class3_bu_edge,
                   ebgcn_lrp_allclass_bu_edge,
                   bigcn_lrp_class0, bigcn_lrp_class1, bigcn_lrp_class2, bigcn_lrp_class3, bigcn_lrp_allclass,
                   ebgcn_lrp_class0, ebgcn_lrp_class1, ebgcn_lrp_class2, ebgcn_lrp_class3, ebgcn_lrp_allclass]
            vec2 = [bigcn_nr_td_fc, bigcn_fr_td_fc, bigcn_tr_td_fc, bigcn_ur_td_fc,
                    bigcn_nr_bu_fc, bigcn_fr_bu_fc, bigcn_tr_bu_fc, bigcn_ur_bu_fc,
                    ebgcn_nr_td_fc, ebgcn_fr_td_fc, ebgcn_tr_td_fc, ebgcn_ur_td_fc,
                    ebgcn_nr_bu_fc, ebgcn_fr_bu_fc, ebgcn_tr_bu_fc, ebgcn_ur_bu_fc]
            vec3 = [bigcn_nr_fc, bigcn_fr_fc, bigcn_tr_fc, bigcn_ur_fc,
                    ebgcn_nr_fc, ebgcn_fr_fc, ebgcn_tr_fc, ebgcn_ur_fc]
            return vec + vec2, vec3

        if datasetname == 'PHEME':
            bigcn_explain_json_path = os.path.join(explain_subdir,
                                                   f'{tree_id}_{split_type}-BiGCNv{versions[1]}_original_explain.json')
            ebgcn_explain_json_path = os.path.join(explain_subdir,
                                                   f'{tree_id}_{split_type}-EBGCNv{versions[2]}_original_explain.json')
        else:
            bigcn_explain_json_path = os.path.join(explain_subdir,
                                                   f'{tree_id}_{split_type}-BiGCNv{lrp_version}_original_explain.json')
            ebgcn_explain_json_path = os.path.join(explain_subdir,
                                                   f'{tree_id}_{split_type}-EBGCNv{lrp_version}_original_explain.json')

        centrality_vec = load_centrality(centrality_json_path, tree_id, error_log)
        # print(centrality_vec)
        if lrp_version != 2:
            original_vec = load_model_explanations(bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log)
        else:
            # try:
            if datasetname == 'PHEME':
                original_vec, bu_td_split = load_model_explanations_twitter1516(bigcn_explain_json_path,
                                                                     ebgcn_explain_json_path, tree_id, error_log)
            else:  # Twitter
                original_vec, bu_td_split = load_model_explanations_twitter1516(bigcn_explain_json_path,
                                                                                ebgcn_explain_json_path, tree_id,
                                                                                error_log)
            if original_vec is None:
                continue
            for i, row in enumerate(bu_td_split):
                if row[0] > row[1]:
                    bu_td_draw_contribution_split[i][0] += 1
                elif row[1] > row[0]:
                    bu_td_draw_contribution_split[i][1] += 1
                else:
                    bu_td_draw_contribution_split[i][2] += 1
            # except:
            #     continue
        if original_vec is None:
            # error_log.append(f'Error: Problem loading original explanation files for Tree num {tree_id}\t')
            continue
        results_vec = original_vec + centrality_vec
        for config in config_type:
            if config == 'edges':
                all_drop_edges_vec = None
                drop_edges_count = 0
                for drop_edges in drop_edge_types:
                    if datasetname == 'PHEME':
                        bigcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-BiGCNv{versions[1]}_d{drop_edges}_explain.json')
                        ebgcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-EBGCNv{versions[2]}_d{drop_edges}_explain.json')
                    else:  # Twitter
                        bigcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-BiGCNv{lrp_version}_d{drop_edges}_explain.json')
                        ebgcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-EBGCNv{lrp_version}_d{drop_edges}_explain.json')
                    if 2 <= lrp_version < 3:
                        # try:
                        if datasetname == 'PHEME':
                            drop_edges_vec, drop_edges_td_bu_split = load_model_explanations_twitter1516(
                                bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log)
                        else:
                            drop_edges_vec, drop_edges_td_bu_split = load_model_explanations_twitter1516(
                                bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log)
                        for i, row in enumerate(drop_edges_td_bu_split):
                            if row[0] > row[1]:
                                drop_edge_bu_td_draw_contribution_split[i][0] += 1
                            elif row[1] > row[0]:
                                drop_edge_bu_td_draw_contribution_split[i][1] += 1
                            else:
                                drop_edge_bu_td_draw_contribution_split[i][2] += 1
                        # except:
                        #     continue
                    else:
                        drop_edges_vec = load_model_explanations(bigcn_explain_json_path, ebgcn_explain_json_path,
                                                                 tree_id, error_log)
                    if drop_edges_vec is None:
                        # error_log.append(f'Error: Problem loading drop edges explanation files for Tree num {tree_id}\t'
                        #                  )
                        continue
                    if all_drop_edges_vec is None:
                        all_drop_edges_vec = drop_edges_vec
                    else:
                        all_drop_edges_vec += drop_edges_vec
                    # print(drop_edges_vec[19:21])
                    drop_edges_count += 1
                if all_drop_edges_vec is None:
                    continue
                results_vec += all_drop_edges_vec
            elif config == 'weights':
                all_initial_weight_vec = None
                initial_weight_count = 0
                for initial_weight_num, initial_weight in enumerate(set_initial_weight_types):
                    if datasetname == 'PHEME':
                        bigcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-BiGCNv{versions[1]}_w{initial_weight}_explain.json')
                        ebgcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-EBGCNv{versions[2]}_w{initial_weight}_explain.json')
                    else:  # Twitter
                        bigcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-BiGCNv{lrp_version}_w{initial_weight}_explain.json')
                        ebgcn_explain_json_path = os.path.join(
                            explain_subdir,
                            f'{tree_id}_{split_type}-EBGCNv{lrp_version}_w{initial_weight}_explain.json')
                    if 2 <= lrp_version < 3:
                        # try:
                        if datasetname == 'PHEME':
                            initial_weight_vec, initial_weight_td_bu_split = load_model_explanations_twitter1516(
                                bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log)
                        else:
                            initial_weight_vec, initial_weight_td_bu_split = load_model_explanations_twitter1516(
                                bigcn_explain_json_path, ebgcn_explain_json_path, tree_id, error_log)
                        for i, row in enumerate(initial_weight_td_bu_split):
                            if row[0] > row[1]:
                                initial_weight_bu_td_draw_contribution_split[initial_weight_num][i][0] += 1
                            elif row[1] > row[0]:
                                initial_weight_bu_td_draw_contribution_split[initial_weight_num][i][1] += 1
                            else:
                                initial_weight_bu_td_draw_contribution_split[initial_weight_num][i][2] += 1
                        # except:
                        #     continue
                    else:
                        initial_weight_vec = load_model_explanations(bigcn_explain_json_path, ebgcn_explain_json_path,
                                                                 tree_id, error_log)
                    if initial_weight_vec is None:
                        # error_log.append(f'Error: Problem loading initial weight explanation files for Tree num {tree_id}\t'
                        #                  )
                        continue
                    if all_initial_weight_vec is None:
                        all_initial_weight_vec = initial_weight_vec
                    else:
                        all_initial_weight_vec += initial_weight_vec
                    # print(drop_edges_vec[19:21])
                    initial_weight_count += 1
                if all_initial_weight_vec is None:
                    continue
                results_vec += all_initial_weight_vec
        # print(results_vec[96])
        temp_mat = np.zeros((1, len(metrics), len(results_vec), len(results_vec)))
        error_flag = False
        for metric_num in range(len(metrics)):
            for i in range(len(results_vec)):
                for j in range(len(results_vec)):
                    try:
                        temp_mat[0, metric_num, i, j] = metrics[metric_num](results_vec[i], results_vec[j])
                    except:
                        # print(tree_id, i, j, len(results_names), len(results_vec))
                        # print(tree_id, i, j, results_names[i], results_names[j], results_vec[i], results_vec[j])
                        temp_mat[0, metric_num, i, j] = -1
                        errors_in_metric[metric_num] += 1
                        errors += 1
                        error_eids.add(tree_id)
                        error_log.append(f'Event: {fold_num}\t'
                                         f'EID: {tree_id}\t'
                                         f'Metric: {metrics[metric_num]}\n'
                                         f'Result 1: {results_names[i]}\t{results_vec[i]}\n'
                                         f'Result 2: {results_names[j]}\t{results_vec[j]}')
                        error_flag = True
        else:
            if error_flag:
                pass
        metrics_mat.append(temp_mat)
        eid_list.append(tree_id)
        total_successful += 1
        # print(temp_mat)
    else:
        if total_num != 0:
            print(f'Dataset: {datasetname}\tSplit: {split_type}\tFold: {event}\tExp method: {exp_method}\n'
                  f'Successfully processed: {total_successful / total_num * 100:.2f}% [{total_successful}/{total_num}]\n')
        else:
            print(f'Dataset: {datasetname}\tSplit: {split_type}\tFold: {event}\tExp method: {exp_method}\n'
                  f'No files to process\n')
        if len(metrics_mat) > 1:
            metrics_mat = np.concatenate(metrics_mat, 0)
        elif len(metrics_mat) == 1:
            metrics_mat = metrics_mat[0]
        else:
            metrics_mat = None
        np_save_filename = ''
        if select == 'quantile':
            np_save_filename = f'q{quantile}'
        elif select == 'interval':
            np_save_filename = f'b{interval}'
        elif select == 'k':
            np_save_filename = f'k{k}'
        if datasetname == 'PHEME':
            np_save_filename += f'_v{versions}'
        for config in config_type:
            if config == 'original':
                np_save_filename += '_original'
            elif config == 'edges':
                np_save_filename += f'_d{drop_edge_types}'
            elif config == 'weights':
                np_save_filename += f'_w{set_initial_weight_types}'
        if exp_method == 'lrp':
            if lrp_version == 2:
                np_save_filename += f'_lrp{lrp_version}'
        else:
            np_save_filename += f'_{exp_method}'
        if save_to_external:
            if metrics_mat is not None:
                np.save(os.path.join(EXTERNAL_EXPLAIN_DIR, f'{datasetname}_{event}_similarity_{np_save_filename}'),
                        metrics_mat)
        else:
            if metrics_mat is not None:
                np.save(os.path.join(EXPLAIN_DIR, f'{datasetname}_{event}_similarity_{np_save_filename}'), metrics_mat)
    for metric_num in range(len(metrics)):
        error_log.append(f'Event: {event}\tMetric: {metrics[metric_num]}\tError: {errors_in_metric[metric_num]}')
    error_log.append(f'Event: {event}\tTotal Errors: {errors}')
    if save_to_external:
        error_log_file_path = os.path.join(EXTERNAL_EXPLAIN_DIR, f'{datasetname}_{event}_similarity_{np_save_filename}'
                                                                 f'_error_log.txt')
    else:
        error_log_file_path = os.path.join(EXPLAIN_DIR, f'{datasetname}_{event}_similarity_{np_save_filename}'
                                                        f'_error_log.txt')
    with open(error_log_file_path, 'w') as f:
        f.write('\n'.join(error_log))
    # print(metrics_mat.shape)
    return metrics_mat, results_names, eid_list, \
           bu_td_draw_contribution_split.tolist(), drop_edge_bu_td_draw_contribution_split.tolist(), \
           [split.tolist() for split in initial_weight_bu_td_draw_contribution_split]


def get_differences_between_distributions(mat, results_names, outputs, loops):
    metric_names = ['jaccard_similarity', 'szymkiewicz_simpson']
    if loops != 0:
        base_len = len(results_names)//loops
    else:
        base_len = len(results_names)
    # outputs = [[50, [66, 67, 68, 69, 70]],
    #            [51, [66, 67, 68, 69, 70]],
    #            [52, [66, 67, 68, 69, 70]],
    #            [53, [66, 67, 68, 69, 70]],
    #            [58, [66, 72, 73, 74, 75]],
    #            [59, [66, 72, 73, 74, 75]],
    #            [60, [66, 72, 73, 74, 75]],
    #            [61, [66, 72, 73, 74, 75]]
    #            ]
    # outputs = [[50, [66, 67, 68, 69, 70]],
    #            [51, [66, 67, 68, 69, 70]],
    #            [52, [66, 67, 68, 69, 70]],
    #            [53, [66, 67, 68, 69, 70]],
    #            [58, [66, 72, 73, 74, 75]],
    #            [59, [66, 72, 73, 74, 75]],
    #            [60, [66, 72, 73, 74, 75]],
    #            [61, [66, 72, 73, 74, 75]],
    #            [54, [66, 67, 68, 69, 70]],
    #            [55, [66, 67, 68, 69, 70]],
    #            [56, [66, 67, 68, 69, 70]],
    #            [57, [66, 67, 68, 69, 70]],
    #            [62, [66, 72, 73, 74, 75]],
    #            [63, [66, 72, 73, 74, 75]],
    #            [64, [66, 72, 73, 74, 75]],
    #            [65, [66, 72, 73, 74, 75]]
    #            ]
    outputs = [[50, [66, 67, 68, 69]],
               [51, [66, 67, 68, 69]],
               [52, [66, 67, 68, 69]],
               [53, [66, 67, 68, 69]],
               [58, [66, 72, 73, 74]],
               [59, [66, 72, 73, 74]],
               [60, [66, 72, 73, 74]],
               [61, [66, 72, 73, 74]],
               [54, [66, 67, 68, 69]],
               [55, [66, 67, 68, 69]],
               [56, [66, 67, 68, 69]],
               [57, [66, 67, 68, 69]],
               [62, [66, 72, 73, 74]],
               [63, [66, 72, 73, 74]],
               [64, [66, 72, 73, 74]],
               [65, [66, 72, 73, 74]]
               ]
    # outputs = [[50, [66, 67, 68, 69, 70]],
    #            [51, [66, 67, 68, 69, 70]],
    #            [52, [66, 67, 68, 69, 70]],
    #            [53, [66, 67, 68, 69, 70]]]
    # outputs = [[54, [66, 67, 68, 69, 70]],
    #            [55, [66, 67, 68, 69, 70]],
    #            [56, [66, 67, 68, 69, 70]],
    #            [57, [66, 67, 68, 69, 70]]]
    # outputs = [[58, [72, 73, 74, 75]],
    #            [59, [72, 73, 74, 75]],
    #            [60, [72, 73, 74, 75]],
    #            [61, [72, 73, 74, 75]]]
    # outputs = [[62, [72, 73, 74, 75]],
    #            [63, [72, 73, 74, 75]],
    #            [64, [72, 73, 74, 75]],
    #            [65, [72, 73, 74, 75]]]
    new_mat = []
    top_labels = []
    row_labels = []
    for metric_num in range(mat.shape[1]):
        new_mat = []
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout(pad=0.5)
        for output_num, (j, output_set) in enumerate(outputs):
            rows = []
            labels = []
            for i in output_set:
                rows.append(mat[:, metric_num, i, j])
                labels.append(f'{results_names[i]}')
            new_mat.append(rows)
            top_labels.append(f'{results_names[j]}')
            row_labels.append(labels)
        new_mat = np.asarray(new_mat)
        num_outputs = len(outputs)
        dist_mat = np.zeros((num_outputs, num_outputs, new_mat.shape[1]))
        ttest_scores = np.zeros((len(row_labels[0]), num_outputs, num_outputs, 2))
        # temp_mat = np.zeros((num_outputs, len(row_labels[0]), new_mat.shape[-1]))
        print(new_mat.shape, ttest_scores.shape)
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[-1]):
                temp1 = np.asarray((np.mean(new_mat[i, j]), np.std(new_mat[i, j]),
                         np.quantile(new_mat[i, j], 0.25), np.quantile(new_mat[i, j], 0.5),
                         np.quantile(new_mat[i, j], 0.75)))
                # print(top_labels[i], row_labels[i][j], temp1)
                # temp_mat[i, j] = new_mat[i, j]
        for centrality_measure in range(ttest_scores.shape[0]):
            for gcn_output1 in range(ttest_scores.shape[1]):
                for gcn_output2 in range(ttest_scores.shape[1]):
                    if gcn_output1 == gcn_output2:
                        continue
                    test_score = stats.ttest_ind(new_mat[gcn_output1, centrality_measure],
                                                 new_mat[gcn_output2, centrality_measure], equal_var=False)

                    ttest_scores[centrality_measure, gcn_output1, gcn_output2] = test_score
                    # print(ttest_scores[gcn_output1, gcn_output2, centrality_measure])
                    # raise Exception
        ttest_scores2 = np.zeros((num_outputs, len(row_labels[0]), len(row_labels[0]), 2))
        for gcn_output in range(ttest_scores2.shape[0]):
            for centrality_measure1 in range(ttest_scores2.shape[1]):
                for centrality_measure2 in range(ttest_scores2.shape[1]):
                    if centrality_measure1 == centrality_measure2:
                        continue
                    test_score = stats.ttest_ind(new_mat[gcn_output, centrality_measure1],
                                                 new_mat[gcn_output, centrality_measure2], equal_var=False)

                    ttest_scores2[gcn_output, centrality_measure1, centrality_measure2] = test_score
        ttest_scores = ttest_scores[:, :, :, -1] >= 0.05
        ttest_scores2 = ttest_scores2[:, :, :, -1] >= 0.05
        with np.printoptions(threshold=np.inf):
            for row in range(ttest_scores.shape[0]):
                print(labels[row], row, ttest_scores[row].sum(0))
                for col in range(ttest_scores.shape[1]):
                    print(row, col, np.nonzero(ttest_scores[row][col]))
                # for col in range(ttest_scores.shape[1]):
                    # print(row, col, ttest_scores[row][0][col])
            # for row in range(ttest_scores2.shape[0]):
            #     print(labels[row], row, ttest_scores2[row].sum(0))
            #     for col in range(ttest_scores2.shape[1]):
            #         print(row, col, np.nonzero(ttest_scores2[row][col]))
            # raise Exception
            # for row in range(ttest_scores.shape[1]):
            #     print(row, ttest_scores[0][row])
            #     raise Exception
        # for i in range(dist_mat.shape[0]):
        #     for j in range(dist_mat.shape[0]):
        #         for k in range(dist_mat.shape[-1]):
        #             if i == j:
        #                 continue
        #             hist1, bin_edges = np.histogram(new_mat[i, k], 100, density=True)
        #             hist2, bin_edges = np.histogram(new_mat[j, k], 100, density=True)
        #             hist1 = hist1 * np.diff(bin_edges)
        #             hist2 = hist2 * np.diff(bin_edges)
        #             # print(i, j, k, special.kl_div(hist1, hist2).sum(),
        #             #       special.rel_entr(hist1, hist2).sum())
        #             temp1 = np.asarray((np.mean(new_mat[i, k]), np.std(new_mat[i, k]),
        #                      np.quantile(new_mat[i, k], 0.25), np.quantile(new_mat[i, k], 0.5),
        #                      np.quantile(new_mat[i, k], 0.75)))
        #             temp2 = np.asarray((np.mean(new_mat[j, k]), np.std(new_mat[j, k]),
        #                      np.quantile(new_mat[j, k], 0.25), np.quantile(new_mat[j, k], 0.5),
        #                      np.quantile(new_mat[j, k], 0.75)))
        #             print(top_labels[i], top_labels[j], k, temp1, temp2)
        #     raise Exception


def summarise_results(mat, results_names, selected_outputs, loops, text_offset=5, font_size=30,
                      vertical=False, fig_save_path=None, save_figs=False):
    metric_names = ['Jaccard', 'Szymkiewicz-Simpson']
    if loops != 0:
        base_len = len(results_names)//loops
    else:
        base_len = len(results_names)
    # new_selected_outputs = []
    # for outputs_row in selected_outputs:
    #     temp_row = []
    #     for loop in range(loops):
    #         for value in outputs_row:
    #             temp_row.append(value + base_len * loop)
    #     new_selected_outputs.append(temp_row)
    # selected_outputs = new_selected_outputs

    if selected_outputs is not None:
        outputs = selected_outputs
    else:
        pass

    # GCN-wise Text
    # outputs = [[0, [5, 10, 15]],
    #            [5, [0, 10, 15]],
    #            [10, [0, 5, 15]],
    #            [15, [0, 5, 10]]]
    # outputs = [[1, [6, 11, 16]],
    #            [6, [1, 11, 16]],
    #            [11, [1, 6, 16]],
    #            [16, [1, 6, 11]]]
    # outputs = [[2, [7, 12, 17]],
    #            [7, [2, 12, 17]],
    #            [12, [2, 7, 17]],
    #            [17, [2, 7, 12]]]
    # outputs = [[3, [8, 13, 18]],
    #            [8, [3, 13, 18]],
    #            [13, [3, 8, 18]],
    #            [18, [3, 8, 13]]]
    # outputs = [[4, [9, 14, 19]],
    #            [9, [4, 14, 19]],
    #            [14, [4, 9, 19]],
    #            [19, [4, 9, 14]]]

    # GCN-wise Edge
    # outputs = [[20, [25, 30, 35]],
    #            [25, [20, 30, 35]],
    #            [30, [20, 25, 35]],
    #            [35, [20, 25, 30]]]
    # outputs = [[21, [26, 31, 36]],
    #            [26, [21, 31, 36]],
    #            [31, [21, 26, 36]],
    #            [36, [21, 26, 31]]]
    # outputs = [[22, [27, 32, 37]],
    #            [27, [22, 32, 37]],
    #            [32, [22, 27, 37]],
    #            [37, [22, 27, 32]]]
    # outputs = [[23, [28, 33, 38]],
    #            [28, [23, 33, 38]],
    #            [33, [23, 28, 38]],
    #            [38, [23, 28, 33]]]
    # outputs = [[24, [29, 34, 39]],
    #            [29, [24, 34, 39]],
    #            [34, [25, 29, 39]],
    #            [39, [24, 29, 34]]]

    # outputs = [[40, [41, 42, 43]],
    #            [41, [40, 42, 43]],
    #            [42, [40, 41, 43]],
    #            [43, [40, 41, 42]]]
    # outputs = [[45, [46, 47, 48]],
    #            [46, [45, 47, 48]],
    #            [47, [45, 46, 48]],
    #            [48, [45, 46, 47]]]

    # Classwise Text
    # outputs = [[0, [1, 2, 3]],
    #            [1, [0, 2, 3]],
    #            [2, [0, 1, 3]],
    #            [3, [0, 1, 2]]]
    # outputs = [[5, [6, 7, 8]],
    #            [6, [5, 7, 8]],
    #            [7, [5, 6, 8]],
    #            [8, [5, 6, 7]]]
    # outputs = [[10, [11, 12, 13]],
    #            [11, [10, 12, 13]],
    #            [12, [10, 11, 13]],
    #            [13, [10, 11, 12]]]
    # outputs = [[15, [16, 17, 18]],
    #            [16, [15, 17, 18]],
    #            [17, [15, 16, 18]],
    #            [18, [15, 16, 17]]]

    # Classwise Edge
    # outputs = [[20, [21, 22, 23]],
    #            [21, [20, 22, 23]],
    #            [22, [20, 21, 23]],
    #            [23, [20, 21, 22]]]
    # outputs = [[25, [26, 27, 28]],
    #            [26, [25, 27, 28]],
    #            [27, [25, 26, 28]],
    #            [28, [25, 26, 27]]]
    # outputs = [[30, [31, 32, 33]],
    #            [31, [30, 32, 33]],
    #            [32, [30, 31, 33]],
    #            [33, [30, 31, 32]]]
    # outputs = [[35, [36, 37, 38]],
    #            [36, [35, 37, 38]],
    #            [37, [35, 36, 38]],
    #            [38, [35, 36, 37]]]

    # outputs = [[0, [50, 51, 52, 53, 55, 57]],
    #            [1, [50, 51, 52, 53, 55, 57]],
    #            [2, [50, 51, 52, 53, 55, 57]],
    #            [3, [50, 51, 52, 53, 55, 57]]]
    # outputs = [[50, [66, 67, 68, 69, 71, 73]],
    #            [51, [66, 67, 68, 69, 71, 73]],
    #            [52, [66, 67, 68, 69, 71, 73]],
    #            [53, [66, 67, 68, 69, 71, 73]]]
    # outputs = [[54, [66, 67, 68, 69, 71, 73]],
    #            [55, [66, 67, 68, 69, 71, 73]],
    #            [56, [66, 67, 68, 69, 71, 73]],
    #            [57, [66, 67, 68, 69, 71, 73]]]

    # outputs = [[50, [66, 67, 68, 69]],
    #            [51, [66, 67, 68, 69]],
    #            [52, [66, 67, 68, 69]],
    #            [53, [66, 67, 68, 69]]]
    # outputs = [[54, [66, 67, 68, 69]],
    #            [55, [66, 67, 68, 69]],
    #            [56, [66, 67, 68, 69]],
    #            [57, [66, 67, 68, 69]]]

    # outputs = [[58, [72, 73, 74, 75]],
    #            [59, [72, 73, 74, 75]],
    #            [60, [72, 73, 74, 75]],
    #            [61, [72, 73, 74, 75]]]
    # outputs = [[62, [66, 72, 73, 74]],
    #            [63, [66, 72, 73, 74]],
    #            [64, [66, 72, 73, 74]],
    #            [65, [66, 73, 74, 75]]]

    # outputs = [[40, [41, 42, 43]],
    #            [41, [40, 42, 43]],
    #            [42, [40, 41, 43]],
    #            [43, [40, 41, 42]]]
    # outputs = [[45, [46, 47, 48]],
    #            [46, [45, 47, 48]],
    #            [47, [45, 46, 48]],
    #            [48, [45, 46, 47]]]

    # outputs = [[0, [76, 142, 208]],
    #            [1, [77, 143, 209]],
    #            [2, [78, 144, 210]],
    #            [3, [79, 145, 211]]]

    # Modified Graph Structure
    # outputs = [[50, [126, 192, 258]],
    #            [51, [127, 193, 259]],
    #            [52, [128, 194, 260]],
    #            [53, [129, 195, 261]]]
    # outputs = [[54, [130, 196, 262]],
    #            [55, [131, 197, 263]],
    #            [56, [132, 198, 264]],
    #            [57, [133, 199, 265]]]
    # outputs = [[58, [134, 200, 266]],
    #            [59, [135, 201, 267]],
    #            [60, [136, 202, 268]],
    #            [61, [137, 203, 269]]]
    # outputs = [[62, [138, 204, 270]],
    #            [63, [139, 205, 271]],
    #            [64, [140, 206, 272]],
    #            [65, [141, 207, 273]]]

    # outputs = [[50, [138, 204, 270, 54, 130, 196, 262]],
    #            [51, [139, 205, 271, 55, 131, 197, 263]],
    #            [52, [140, 206, 272, 56, 132, 198, 264]],
    #            [53, [141, 207, 273, 57, 133, 199, 265]]]
    # outputs = [[66, [70]]]
    # outputs = [[40, [50, 51, 52, 53, 55, 57]],
    #            [41, [50, 51, 52, 53, 55, 57]],
    #            [42, [50, 51, 52, 53, 55, 57]],
    #            [43, [50, 51, 52, 53, 55, 57]]]
    # outputs = [[45, [50, 51, 52, 53, 55, 57]],
    #            [46, [50, 51, 52, 53, 55, 57]],
    #            [47, [50, 51, 52, 53, 55, 57]],
    #            [48, [50, 51, 52, 53, 55, 57]]]
    # outputs = [[0, [82, 83, 84, 85, 87, 89]],
    #            [1, [82, 83, 84, 85, 87, 89]],
    #            [2, [82, 83, 84, 85, 87, 89]],
    #            [3, [82, 83, 84, 85, 87, 89]],
    #            ]

    # Construct inputs
    if vertical:
        fig, ax = plt.subplots(len(outputs), 1, figsize=(12, 24))
        fig.tight_layout(pad=2.0)
        fig: plt.Figure
        fig.subplots_adjust(left=0.1, right=0.98, wspace=0.1, hspace=0.25)
    else:
        fig, ax = plt.subplots(len(outputs)//2, 2, figsize=(12, 12))
        fig.tight_layout(pad=2.0)
        fig: plt.Figure
        fig.subplots_adjust(left=0.05, right=0.98, wspace=0.1, hspace=0.25)
    palette = ['paleturquoise', 'yellow']
    for output_num, (j, output_set) in enumerate(outputs):
        rows = []
        labels = []
        column_names = ['score', 'metric', 'name']
        means, medians = [], []
        uqs, lqs = [], []
        std = []
        for i in output_set:
            mean1 = mat[:, 0, i, j].mean()
            mean2 = mat[:, 1, i, j].mean()
            median1 = np.median(mat[:, 0, i, j])
            median2 = np.median(mat[:, 1, i, j])
            lq1 = np.quantile(mat[:, 0, i, j], 0.25)
            lq2 = np.quantile(mat[:, 1, i, j], 0.25)
            uq1 = np.quantile(mat[:, 0, i, j], 0.75)
            uq2 = np.quantile(mat[:, 1, i, j], 0.75)
            std1 = mat[:, 0, i, j].std()  # For table stats
            std2 = mat[:, 1, i, j].std()
            row = np.concatenate((mat[:, 0, i, j], mat[:, 1, i, j]))
            temp1 = ['Jaccard'] * (row.shape[0]//2)
            temp2 = ['Szymkiewicz-Simpson'] * (row.shape[0]//2)
            temp_metric = temp1 + temp2
            temp_score = row.tolist()
            # temp_name = [f'{results_names[i][:-5]}'] * row.shape[0]
            # temp_name = [f'{results_names[i][:-8]}'] * row.shape[0]
            # temp_name = [f'{results_names[i][:-3]}'] * row.shape[0]
            if text_offset != 0:
                temp_name = [f'{results_names[i][:-text_offset]}'] * row.shape[0]
            else:
                temp_name = [f'{results_names[i]}'] * row.shape[0]
            temp_row = [temp_score, temp_metric, temp_name]
            row = np.asarray(temp_row).transpose()
            rows.append(row)
            labels.append(f'{results_names[i]}')
            means += [mean1, mean2]
            medians += [median1, median2]
            uqs += [uq1, uq2]
            lqs += [lq1, lq2]
            std += [std1, std2]
            # print(row.shape)
            print(f'{results_names[j]} VS {results_names[i]}:\t'
                  f'{metric_names[0]}: Mean: {mean1:.3f} Std: {std1:.2f}\t'
                  f'{metric_names[1]}: Mean: {mean2:.3f} Std: {std2:.2f}\n')
        rows = np.concatenate(rows)
        # print(rows.shape)
        new_frame = pd.DataFrame(rows, columns=column_names)
        new_frame['score'] = pd.to_numeric(new_frame['score'])
        # print(new_frame.head())
        # print(new_frame.tail())
        # print(new_frame.iloc[1, 0], type(new_frame.iloc[1, 0]))
        # print(new_frame['metric'].unique(), new_frame['name'].unique())
        if vertical:
            axs = ax[output_num]
        else:
            axs = ax[output_num // 2, output_num % 2]
        axs.set_yticks([1.0, 0.75, 0.5, 0.25, 0.0])
        axs.grid(True, axis='y', alpha=0.5)
        axs: plt.Axes = sns.violinplot(new_frame, x='name', y='score', hue='metric', split=True, gridsize=1000,
                                       cut=0, inner=None, palette=palette,
                                       ax=axs)
        xticks_pos = axs.get_xticks()

        for pos_num, pos in enumerate(xticks_pos):
            # left half of violin
            new_pos = [pos, pos - .45]
            new_means = [means[pos_num * 2]] * 2
            new_medians = [medians[pos_num * 2]] * 2
            new_uqs = [uqs[pos_num * 2]] * 2
            new_lqs = [lqs[pos_num * 2]] * 2
            axs.plot(new_pos, new_means, 'r-', alpha=0.6)
            axs.plot(new_pos, new_medians, 'k--', alpha=0.6)
            axs.plot(new_pos, new_uqs, 'g--', alpha=0.6)
            axs.plot(new_pos, new_lqs, 'b--', alpha=0.6)
            # right half of violin
            new_pos = [pos, pos + .45]
            new_means = [means[pos_num * 2 + 1]] * 2
            new_medians = [medians[pos_num * 2 + 1]] * 2
            new_uqs = [uqs[pos_num * 2 + 1]] * 2
            new_lqs = [lqs[pos_num * 2 + 1]] * 2
            if pos_num != len(xticks_pos) - 1:
                axs.plot(new_pos, new_means, 'r-', alpha=0.6)
                axs.plot(new_pos, new_medians, 'k--', alpha=0.6)
                axs.plot(new_pos, new_uqs, 'g--', alpha=0.6)
                axs.plot(new_pos, new_lqs, 'b--', alpha=0.6)
                axs.legend()
            else:
                axs.plot(new_pos, new_means, 'r-', alpha=0.6, label='Mean')
                axs.plot(new_pos, new_medians, 'k--', alpha=0.6, label='Median')
                axs.plot(new_pos, new_uqs, 'g--', alpha=0.6, label='Upper Quartile')
                axs.plot(new_pos, new_lqs, 'b--', alpha=0.6, label='Lower Quartile')
                axs.legend()
        # axs.set_title(f'{results_names[j][:-5]} VS', fontsize=20)
        # axs.set_title(f'{results_names[j][:-8]} VS', fontsize=20)
        # axs.set_title(f'{results_names[j][:-3]} VS', fontsize=20)
        if text_offset != 0:
            axs.set_title(f'{results_names[j][:-text_offset]} VS', fontsize=font_size)
        else:
            axs.set_title(f'{results_names[j]} VS', fontsize=font_size)
        axs.tick_params(axis='x', which='major', labelsize=font_size)
        axs.tick_params(axis='y', which='major', labelsize=20)
        # axs.set_ylabel('score', fontsize=20)
        # if output_num % 2 != 1:
        #     axs.set_ylabel('score', fontsize=20)
        # else:
        #     axs.set_ylabel('', fontsize=0)
        axs.set_ylabel('')
        axs.set_xlabel('')
        # axs.get_legend().remove()
        if vertical:
            legend = axs.legend(framealpha=0.6)
            for text in legend.texts:
                text.set_alpha(0.8)
            # if output_num != 0:
            #     axs.get_legend().remove()
            # else:
            #     axs.legend(bbox_to_anchor=(0, -0.1))
        else:
            if output_num != len(outputs) - 1:
                axs.get_legend().remove()
            else:
                axs.legend(bbox_to_anchor=(0, -0.1))
    # figManager = plt.get_current_fig_manager()
    # figManager.window.state('zoomed')
    # plt.show()
    if save_figs:
        if fig_save_path is not None:
            plt.savefig(fig_save_path, format='pdf')
        else:
            plt.savefig('figure.pdf', format='pdf')

    # raise Exception

    # for metric_num in range(mat.shape[1]):
    #     fig, axs = plt.subplots(2, 2)
    #     fig.tight_layout(pad=0.5)
    #     for output_num, (j, output_set) in enumerate(outputs):
    #         rows = []
    #         labels = []
    #         for i in output_set:
    #             rows.append(mat[:, metric_num, i, j])
    #             labels.append(f'{results_names[i]}')
    #             print(f'Metric: {metric_names[metric_num]}\tResults: {results_names[i]} v {results_names[j]}\n'
    #                   f'Mean: {np.mean(mat[:, metric_num, i, j])}\tStd: {np.std(mat[:, metric_num, i, j])}')
    #         axs[output_num // 2, output_num % 2].set(title=f'{metric_names[metric_num]} for {results_names[j]}')
    #         parts = axs[output_num // 2, output_num % 2].violinplot(rows, quantiles=[[0.25, 0.75] for row in rows],
    #                                                                 points=1000, showmeans=True, showmedians=True,
    #                                                                 showextrema=True)
    #         colours = ['red']
    #         parts['cmeans'].set_color(colours)
    #         colours = ['green']
    #         parts['cmedians'].set_color(colours)
    #         colours = ['purple']
    #         parts['cquantiles'].set_color(colours)
    #         axs[output_num // 2, output_num % 2].set_xticks([i + 1 for i in range(len(rows))],
    #                                                         rotation=0, labels=labels, fontsize=8)
    #         axs[output_num // 2, output_num % 2].set_yticks(np.linspace(0, 1, 5))
    #         figManager = plt.get_current_fig_manager()
    #         figManager.window.state('zoomed')

    # for metric_num in range(mat.shape[1]):
    #     fig, axs = plt.subplots(2, 2)
    #     fig.tight_layout(pad=0.5)
    #     for output_num, (j, output_set) in enumerate(outputs):
    #         rows = []
    #         labels = []
    #         for i in output_set:
    #             rows.append(mat[:, metric_num, i, j])
    #             labels.append(f'{results_names[i]}')
    #             print(f'Metric: {metric_names[metric_num]}\tResults: {results_names[i]} v {results_names[j]}\n'
    #                   f'Mean: {np.mean(mat[:, metric_num, i, j])}\tStd: {np.std(mat[:, metric_num, i, j])}')
    #         axs[output_num // 2, output_num % 2].set(title=f'{metric_names[metric_num]} for {results_names[j]}')
    #         parts = axs[output_num // 2, output_num % 2].violinplot(rows, quantiles=[[0.25, 0.75] for row in rows],
    #                                                                 points=1000, showmeans=True, showmedians=True,
    #                                                                 showextrema=True)
    #         colours = ['red']
    #         parts['cmeans'].set_color(colours)
    #         colours = ['green']
    #         parts['cmedians'].set_color(colours)
    #         colours = ['purple']
    #         parts['cquantiles'].set_color(colours)
    #         axs[output_num // 2, output_num % 2].set_xticks([i + 1 for i in range(len(rows))],
    #                                                         rotation=0, labels=labels, fontsize=8)
    #         axs[output_num // 2, output_num % 2].set_yticks(np.linspace(0, 1, 5))
    #         figManager = plt.get_current_fig_manager()
    #         figManager.window.state('zoomed')

    # for output_set in selected_outputs:
    #     for metric_num in range(mat.shape[1]):
    #         # fig, axs = plt.subplots(len(output_set), 2)
    #         fig, axs = plt.subplots((len(output_set)//2) + 1 if len(output_set) % 2 != 0 else len(output_set)//2, 2)
    #         # print(len(output_set)//2)
    #         # raise Exception
    #         fig.tight_layout(pad=0.5)
    #         for output_num, i in enumerate(output_set):
    #             rows = []
    #             labels = []
    #             for j in output_set:
    #                 if i != j:
    #                     rows.append(mat[:, metric_num, i, j])
    #                     labels.append(f'{results_names[j]}')
    #                     print(f'Metric: {metric_names[metric_num]}\tResults: {results_names[i]} v {results_names[j]}\n'
    #                           f'Mean: {np.mean(mat[:, metric_num, i, j])}\tStd: {np.std(mat[:, metric_num, i, j])}')
    #             axs[output_num//2, output_num % 2].set(title=f'{metric_names[metric_num]} for {results_names[i]}')
    #             parts = axs[output_num//2, output_num % 2].violinplot(rows, quantiles=[[0.25, 0.75] for row in rows],
    #                                                                   points=1000, showmeans=True, showmedians=True,
    #                                                                   showextrema=True)
    #             # colours = ['red', 'red', 'red', 'red']
    #             colours = ['red']
    #             parts['cmeans'].set_color(colours)
    #             colours = ['green']
    #             parts['cmedians'].set_color(colours)
    #             colours = ['purple']
    #             parts['cquantiles'].set_color(colours)
    #             axs[output_num//2, output_num % 2].set_xticks([i + 1 for i in range(len(rows))],
    #                                                           rotation=0, labels=labels, fontsize=8)
    #             axs[output_num//2, output_num % 2].set_yticks(np.linspace(0, 1, 5))
    #             # With histogram
    #             # axs[output_num, 0].set(title=f'{metric_names[metric_num]} for {vec[i]}')
    #             # axs[output_num, 0].violinplot(rows)
    #             # axs[output_num, 0].set_xticks([i + 1 for i in range(len(rows))], labels=labels, fontsize=8)
    #             # axs[output_num, 0].set_yticks(np.linspace(0, 1, 5))
    #             # axs[output_num, 1].hist(rows, bins=np.linspace(0, 1, 11))
    #             # axs[output_num, 1].set_xticks(np.linspace(0, 1, 11))
    #         figManager = plt.get_current_fig_manager()
    #         figManager.window.state('zoomed')
    # plt.show()


def plot_dummy_figure(mat, results_names, selected_outputs, loops):
    metric_names = ['jaccard_similarity', 'szymkiewicz_simpson']
    outputs = [[62, [138, 204, 270]],
               [63, [139, 205, 271]],
               [64, [140, 206, 272]],
               [65, [141, 207, 273]]]
    fig, ax = plt.subplots(4, 1)
    # fig, ax = plt.subplots(1)
    fig.tight_layout(pad=2.0)
    fig: plt.Figure
    fig.subplots_adjust(left=0.05, right=0.98, wspace=0.1, hspace=0.25)
    palette = ['paleturquoise', 'yellow']
    mat = np.random.rand(100, 2, 274, 274)  # dummy
    for output_num, (j, output_set) in enumerate(outputs):
        rows = []
        labels = []
        column_names = ['score', 'metric', 'name']
        means, medians = [], []
        uqs, lqs = [], []
        for i in output_set:
            mean1 = mat[:, 0, i, j].mean()
            mean2 = mat[:, 1, i, j].mean()
            median1 = np.median(mat[:, 0, i, j])
            median2 = np.median(mat[:, 1, i, j])
            lq1 = np.quantile(mat[:, 0, i, j], 0.25)
            lq2 = np.quantile(mat[:, 1, i, j], 0.25)
            uq1 = np.quantile(mat[:, 0, i, j], 0.75)
            uq2 = np.quantile(mat[:, 1, i, j], 0.75)
            row = np.concatenate((mat[:, 0, i, j], mat[:, 1, i, j]))
            temp1 = ['Jaccard'] * (row.shape[0] // 2)
            temp2 = ['Szymkiewicz-Simpson'] * (row.shape[0] // 2)
            temp_metric = temp1 + temp2
            temp_score = row.tolist()
            # temp_name = [f'{results_names[i][:-5]}'] * row.shape[0]
            # temp_name = [f'{results_names[i][:-8]}'] * row.shape[0]
            temp_name = [f'{results_names[i][:-3]}'] * row.shape[0]
            temp_row = [temp_score, temp_metric, temp_name]
            row = np.asarray(temp_row).transpose()
            rows.append(row)
            labels.append(f'{results_names[i]}')
            means += [mean1, mean2]
            medians += [median1, median2]
            uqs += [uq1, uq2]
            lqs += [lq1, lq2]
            # print(row.shape)
        rows = np.concatenate(rows)
        # print(rows.shape)
        new_frame = pd.DataFrame(rows, columns=column_names)
        new_frame['score'] = pd.to_numeric(new_frame['score'])
        # print(new_frame.head())
        # print(new_frame.tail())
        # print(new_frame.iloc[1, 0], type(new_frame.iloc[1, 0]))
        # print(new_frame['metric'].unique(), new_frame['name'].unique())

        # axs = ax[output_num // 2, output_num % 2]
        axs = ax[output_num]
        # axs = ax
        axs.set_yticks([1.0, 0.75, 0.5, 0.25, 0.0])
        axs.grid(True, axis='y', alpha=0.5)
        axs: plt.Axes = sns.violinplot(new_frame, x='name', y='score', hue='metric', split=True, gridsize=1000,
                                       cut=0, inner=None, palette=palette,
                                       ax=axs, alpha=0.4)
        alpha = 0.2
        for violin in axs.collections:
            violin.set_alpha(alpha)
        # xticks_pos = axs.get_xticks()
        axs.set_title(f'{results_names[j][:-3]} VS', fontsize=20)
        axs.tick_params(axis='x', which='major', labelsize=20)
        axs.tick_params(axis='y', which='major', labelsize=18)
        # axs.set_ylabel('score', fontsize=20)
        # if output_num % 2 != 1:
        #     axs.set_ylabel('score', fontsize=20)
        # else:
        #     axs.set_ylabel('', fontsize=0)
        axs.set_ylabel('')
        axs.set_xlabel('')
        if output_num != len(outputs) - 1:
            axs.get_legend().remove()
        else:
            axs.legend(bbox_to_anchor=(0, -0.1))
        # axs.get_legend().remove()
    plt.show()


if __name__ == '__main__':
    datasetname = 'PHEME'  # ['Twitter', 'PHEME']
    split_type = '9fold'  # ['5fold', '9fold']
    versions = [2.1, 2.1, 2.1]
    lrp_version = 2
    config_type = ['edges', 'weights']  # ['original', 'nodes', 'edges', 'weights', 'centrality']
    select = 'quantile'  # 'quantile', 'interval', 'k'
    quantile = 0.75
    interval = 0.75
    k = 10
    min_graph_size = 20
    randomise_types = [0.25]  # [1.0, 0.75, 0.5, 0.25]
    drop_edge_types = [1]  # [1.0, 0.75, 0.5, 0.25]
    set_initial_weight_types = [0.5, 2.0]  # [0.5, 1.0, 2.0]
    save_to_external = False
    exp_method = 'lrp'  # ['lrp', 'cam', 'eb']
    args = {'datasetname': datasetname,
            'event': 'fold0',
            'split_type': split_type,
            'versions': versions,
            'lrp_version': lrp_version,
            'config_type': config_type,
            'select': select,
            'quantile': quantile,
            'interval': interval,
            'k': k,
            'min_graph_size': min_graph_size,
            'randomise_types': randomise_types,
            'drop_edge_types': drop_edge_types,
            'set_initial_weight_types': set_initial_weight_types,
            'save_to_external': save_to_external,
            'exp_method': exp_method}
    vec = ['bigcn_nr_td_text', 'bigcn_fr_td_text', 'bigcn_tr_td_text', 'bigcn_ur_td_text',
           'bigcn_all_td_text',
           'ebgcn_nr_td_text', 'ebgcn_fr_td_text', 'ebgcn_tr_td_text', 'ebgcn_ur_td_text',
           'ebgcn_all_td_text',
           'bigcn_nr_bu_text', 'bigcn_fr_bu_text', 'bigcn_tr_bu_text', 'bigcn_ur_bu_text',
           'bigcn_all_bu_text',
           'ebgcn_nr_bu_text', 'ebgcn_fr_bu_text', 'ebgcn_tr_bu_text', 'ebgcn_ur_bu_text',
           'ebgcn_all_bu_text',
           'bigcn_nr_td_edge', 'bigcn_fr_td_edge', 'bigcn_tr_td_edge', 'bigcn_ur_td_edge',
           'bigcn_all_td_edge',
           'ebgcn_nr_td_edge', 'ebgcn_fr_td_edge', 'ebgcn_tr_td_edge', 'ebgcn_ur_td_edge',
           'ebgcn_all_td_edge',
           'bigcn_nr_bu_edge', 'bigcn_fr_bu_edge', 'bigcn_tr_bu_edge', 'bigcn_ur_bu_edge',
           'bigcn_all_bu_edge',
           'ebgcn_nr_bu_edge', 'ebgcn_fr_bu_edge', 'ebgcn_tr_bu_edge', 'ebgcn_ur_bu_edge',
           'ebgcn_all_bu_edge',
           'bigcn_nr_joint', 'bigcn_fr_joint', 'bigcn_tr_joint', 'bigcn_ur_joint', 'bigcn_all_joint',
           'ebgcn_nr_joint', 'ebgcn_fr_joint', 'ebgcn_tr_joint', 'ebgcn_ur_joint', 'ebgcn_all_joint']
    centrality = ['td_out_degree', 'td_betweenness', 'td_closeness', 'td_farness', 'td_eigencentrality',
                  'bu_out_degree', 'bu_betweenness', 'bu_closeness', 'bu_farness', 'bu_eigencentrality']
    vec2 = ['bigcn_nr_td', 'bigcn_fr_td', 'bigcn_tr_td', 'bigcn_ur_td',
            'ebgcn_nr_td', 'ebgcn_fr_td', 'ebgcn_tr_td', 'ebgcn_ur_td',
            'bigcn_nr_bu', 'bigcn_fr_bu', 'bigcn_tr_bu', 'bigcn_ur_bu',
            'ebgcn_nr_bu', 'ebgcn_fr_bu', 'ebgcn_tr_bu', 'ebgcn_ur_bu']
    vec3 = ['bigcn_nr_fc', 'bigcn_fr_fc', 'bigcn_tr_fc', 'bigcn_ur_fc',
            'ebgcn_nr_fc', 'ebgcn_fr_fc', 'ebgcn_tr_fc', 'ebgcn_ur_fc']
    vec += vec2
    vec += centrality
    # dummy_vec = []
    # temp_vec = []
    # for v in vec:
    #     temp_vec += [v]
    # for v in vec2:
    #     temp_vec += [v]
    # for temp_v in temp_vec:
    #     dummy_vec += [temp_v]
    # for temp_v in temp_vec:
    #     dummy_vec += [f'{temp_v}_d1']
    # for temp_v in temp_vec:
    #     dummy_vec += [f'{temp_v}_w0.5']
    # for temp_v in temp_vec:
    #     dummy_vec += [f'{temp_v}_w2.0']
    # for temp_c in centrality:
    #     dummy_vec += [temp_c]
    # plot_dummy_figure([], dummy_vec, [], 0)
    # raise Exception

    # [0, 1, 2, 3,
    #  4, bigcn td text
    #  5, 6, 7, 8,
    #  9, ebgcn td text
    #  10, 11, 12, 13,
    #  14, bigcn bu text
    #  15, 16, 17, 18,
    #  19, ebgcn bu text
    #  20, 21, 22, 23,
    #  24, bigcn td edge
    #  25, 26, 27, 28,
    #  29, ebgcn td edge
    #  30, 31, 32, 33,
    #  34, bigcn bu edge
    #  35, 36, 37, 38
    #  39, ebgcn bu edge
    #  40, 41, 42, 43, 44, bigcn joint
    #  45, 46, 47, 48, 49, ebgcn joint
    #  ]
    # [50, 51, 52, 53, bigcn td
    #  54, 55, 56, 57, ebgcn td
    #  58, 59, 60, 61, bigcn bu
    #  62, 63, 64, 65, ebgcn bu
    #  ]
    #  [66, 67, 68, 69, 70, td centrality, deg, bet, close, far, eigen
    #  71, 72, 73, 74, 75 bu centrality
    #  ]

    # selected_outputs = [[0, 1, 2, 3],
    #                     [5, 6, 7, 8],
    #                     [10, 11, 12, 13],
    #                     [15, 16, 17, 18],
    #                     [20, 21, 22, 23],
    #                     [25, 26, 27, 28],
    #                     [30, 31, 32, 33],
    #                     [35, 36, 37, 38]]
    # selected_outputs = [[0, 1, 2, 3, 5, 6, 7, 8],
    #                     [10, 11, 12, 13, 15, 16, 17, 18]]
    # selected_outputs = [[0, 5, 10, 15],
    #                     [20, 25, 30, 35]]
    # selected_outputs = [[0, 5, 10, 15],
    #                     [20, 25, 30, 35]]  # nr
    # selected_outputs = [[1, 6, 11, 16],
    #                     [21, 26, 31, 36]]  # fr
    # selected_outputs = [[2, 7, 12, 17],
    #                     [22, 27, 32, 37]]  # tr
    # selected_outputs = [[3, 8, 13, 18],
    #                     [23, 28, 33, 38]]  # ur
    # selected_outputs = [[0, 1, 2, 3],
    #                     [5, 6, 7, 8],
    #                     [10, 11, 12, 13],
    #                     [15, 16, 17, 18]]  # text
    # selected_outputs = [[20, 21, 22, 23],
    #                     [25, 26, 27, 28],
    #                     [30, 31, 32, 33],
    #                     [35, 36, 37, 38]]  # edge

    np_save_filename = ''
    if select == 'quantile':
        np_save_filename = f'q{quantile}'
    elif select == 'interval':
        np_save_filename = f'b{interval}'
    elif select == 'k':
        np_save_filename = f'k{k}'
    if datasetname == 'PHEME':
        np_save_filename += f'_v{versions}'
    for config in config_type:
        if config == 'original':
            np_save_filename += '_original'
        elif config == 'edges':
            np_save_filename += f'_d{drop_edge_types}'
        elif config == 'weights':
            np_save_filename += f'_w{set_initial_weight_types}'
    if exp_method == 'lrp':
        if lrp_version == 2:
            np_save_filename += f'_lrp{lrp_version}'
    else:
        np_save_filename += f'_{exp_method}'
    try:
        if save_to_external:
            results_mat = np.load(os.path.join(EXTERNAL_EXPLAIN_DIR,
                                               f'{datasetname}_{split_type}_{np_save_filename}.npy'))
            with open(os.path.join(EXTERNAL_EXPLAIN_DIR, f'{datasetname}_{split_type}_{np_save_filename}_resultsnames.json'),
                      'r') as json_file:
                results_names = json.load(json_file)
            with open(os.path.join(EXTERNAL_EXPLAIN_DIR, f'{datasetname}_{split_type}_{np_save_filename}_eids.json'),
                      'r') as json_file:
                fold_eids = json.load(json_file)
            with open(
                    os.path.join(EXTERNAL_EXPLAIN_DIR, f'{datasetname}_{split_type}_{np_save_filename}_contribution_split.json'),
                    'r') as json_file:
                temp = json.load(json_file)
                contribution_split = temp['original']
                drop_edge_contribution_split = temp['drop_edge']
                initial_weight_contribution_split = temp['initial_weight']
        else:
            results_mat = np.load(os.path.join(EXPLAIN_DIR, f'{datasetname}_{split_type}_similarity_{np_save_filename}.npy'))
        # raise Exception
            with open(os.path.join(EXPLAIN_DIR, f'{datasetname}_{split_type}_similarity_{np_save_filename}_resultsnames.json'),
                      'r') as json_file:
                results_names = json.load(json_file)
            with open(os.path.join(EXPLAIN_DIR, f'{datasetname}_{split_type}_similarity_{np_save_filename}_eids.json'),
                      'r') as json_file:
                fold_eids = json.load(json_file)
            with open(os.path.join(EXPLAIN_DIR, f'{datasetname}_{split_type}_similarity_{np_save_filename}_contribution_split.json'),
                      'r') as json_file:
                temp = json.load(json_file)
                contribution_split = temp['original']
                drop_edge_contribution_split = temp['drop_edge']
                initial_weight_contribution_split = temp['initial_weight']
    except Exception as e:
        raise Exception
        # print(e)
        results_mat = []
        results_names = []
        fold_eids = []
        if datasetname == 'PHEME':
            if split_type == '5fold':
                for fold_num in range(5):
                    event = f'fold{fold_num}'
                    args['event'] = event
                    results, results_names, eids, contribution_split, drop_edge_contribution_split, \
                    initial_weight_contribution_split = \
                        compute_similarities_model_attribution_nfold(args)
                    results_mat.append(results)
                    fold_eids += eids
                    # print(results)
            elif split_type == '9fold':  # Eventwise split
                for fold_num in range(9):
                    event = f'{FOLD_2_EVENTNAME[fold_num]}'  # Get Event name from event number
                    args['event'] = event
                    results, results_names, eids, contribution_split, drop_edge_contribution_split, \
                    initial_weight_contribution_split = \
                        compute_similarities_model_attribution_nfold(args)
                    save_path = os.path.join(EXPLAIN_DIR, f'{datasetname}_{event}_results_{np_save_filename}.txt')
                    with open(save_path, 'w') as f:
                        f.write(f'{results}')
                    if results is not None:
                        results_mat.append(results)
                        fold_eids += eids
                    else:
                        print(f'No results for {event}\n')
        else:  # Twitter
            for fold_num in range(5):
                event = f'fold{fold_num}'
                args['event'] = event
                results, results_names, eids, contribution_split, drop_edge_contribution_split, \
                initial_weight_contribution_split = \
                    compute_similarities_model_attribution_nfold(args)
                save_path = os.path.join(EXPLAIN_DIR, f'{datasetname}_{event}_results_{np_save_filename}.txt')
                with open(save_path, 'w') as f:
                    f.write(f'{results}')
                if results is not None:
                    results_mat.append(results)
                    fold_eids += eids
                else:
                    print(f'No results for {event}\n')
        # temp_results_mat, temp_results_names = [], []
        # for i in range(len(results_mat)):
        # print(results_mat)
        results_mat = np.concatenate(results_mat, 0)
        if save_to_external:
            np.save(os.path.join(EXTERNAL_EXPLAIN_DIR, f'{datasetname}_{split_type}_{np_save_filename}'), results_mat)
            with open(os.path.join(EXTERNAL_EXPLAIN_DIR,
                                   f'{datasetname}_{split_type}_{np_save_filename}_resultsnames.json'),
                      'w') as json_file:
                json.dump(results_names, json_file, indent=1)
            with open(os.path.join(EXTERNAL_EXPLAIN_DIR, f'{datasetname}_{split_type}_{np_save_filename}_eids.json'),
                      'w') as json_file:
                json.dump(fold_eids, json_file, indent=1)
            with open(
                    os.path.join(EXTERNAL_EXPLAIN_DIR,
                                 f'{datasetname}_{split_type}_{np_save_filename}_contribution_split.json'),
                    'w') as json_file:
                json.dump({'original': contribution_split,
                           'drop_edge': drop_edge_contribution_split,
                           'initial_weight': initial_weight_contribution_split},
                          json_file, indent=1)
        else:
            np.save(os.path.join(EXPLAIN_DIR, f'{datasetname}_{split_type}_similarity_{np_save_filename}'), results_mat)
            with open(os.path.join(EXPLAIN_DIR, f'{datasetname}_{split_type}_similarity_{np_save_filename}'
                                                f'_resultsnames.json'),
                      'w') as json_file:
                json.dump(results_names, json_file, indent=1)
            with open(os.path.join(EXPLAIN_DIR, f'{datasetname}_{split_type}_similarity_{np_save_filename}_eids.json'),
                      'w') as json_file:
                json.dump(fold_eids, json_file, indent=1)
            with open(os.path.join(EXPLAIN_DIR,
                                   f'{datasetname}_{split_type}_similarity_{np_save_filename}_contribution_split.json'),
                      'w') as json_file:
                json.dump({'original': contribution_split,
                           'drop_edge': drop_edge_contribution_split,
                           'initial_weight': initial_weight_contribution_split},
                          json_file, indent=1)
    loops = len(results_names) // len(vec)
    print(results_mat.shape, len(results_names), len(vec), loops)
    print(contribution_split)
    print(drop_edge_contribution_split)
    for split in initial_weight_contribution_split:
        print(split)
    save_figs = True
    fig_counter = 0
    selected_outputs_list = [
        [
            [4, [9, 14, 19]],
            [9, [4, 14, 19]],
            [14, [4, 9, 19]],
            [19, [4, 9, 14]]
        ],
        [
            [24, [29, 34, 39]],
            [29, [24, 34, 39]],
            [34, [25, 29, 39]],
            [39, [24, 29, 34]]
        ],
        [
            [0, [1, 2, 3]],
            [1, [0, 2, 3]],
            [2, [0, 1, 3]],
            [3, [0, 1, 2]]
        ],
        [
            [5, [6, 7, 8]],
            [6, [5, 7, 8]],
            [7, [5, 6, 8]],
            [8, [5, 6, 7]]
        ],
        [
            [10, [11, 12, 13]],
            [11, [10, 12, 13]],
            [12, [10, 11, 13]],
            [13, [10, 11, 12]]
        ],
        [
            [15, [16, 17, 18]],
            [16, [15, 17, 18]],
            [17, [15, 16, 18]],
            [18, [15, 16, 17]]
        ],
        [
            [20, [21, 22, 23]],
            [21, [20, 22, 23]],
            [22, [20, 21, 23]],
            [23, [20, 21, 22]]
        ],
        [
            [25, [26, 27, 28]],
            [26, [25, 27, 28]],
            [27, [25, 26, 28]],
            [28, [25, 26, 27]]],
        [
            [30, [31, 32, 33]],
            [31, [30, 32, 33]],
            [32, [30, 31, 33]],
            [33, [30, 31, 32]]
        ],
        [
            [35, [36, 37, 38]],
            [36, [35, 37, 38]],
            [37, [35, 36, 38]],
            [38, [35, 36, 37]]
        ]
    ]
    for fig_num, selected_outputs in enumerate(selected_outputs_list):
        fig_save_path = f'figure_{fig_counter}.pdf'
        summarise_results(results_mat, results_names, selected_outputs, loops, text_offset=5, font_size=24,
                          vertical=True, fig_save_path=fig_save_path, save_figs=save_figs)
        fig_counter += 1
    selected_outputs_list = [
        [
            [50, [126, 192, 258]],
            [51, [127, 193, 259]],
            [52, [128, 194, 260]],
            [53, [129, 195, 261]]
        ],
        [
            [54, [130, 196, 262]],
            [55, [131, 197, 263]],
            [56, [132, 198, 264]],
            [57, [133, 199, 265]]
        ],
        [
            [58, [134, 200, 266]],
            [59, [135, 201, 267]],
            [60, [136, 202, 268]],
            [61, [137, 203, 269]]
        ],
        [
            [62, [138, 204, 270]],
            [63, [139, 205, 271]],
            [64, [140, 206, 272]],
            [65, [141, 207, 273]]]
    ]
    for fig_num, selected_outputs in enumerate(selected_outputs_list):
        fig_save_path = f'figure_{fig_counter}.pdf'
        summarise_results(results_mat, results_names, selected_outputs, loops, text_offset=0, font_size=24,
                          vertical=True, fig_save_path=fig_save_path, save_figs=save_figs)
        fig_counter += 1
    if save_figs:
        print(f'Figures generated: {fig_counter}')
    # summarise_results(results_mat, results_names, selected_outputs, loops, vertical=True)
    # get_differences_between_distributions(results_mat, results_names, selected_outputs, loops)
    print('End of programme')