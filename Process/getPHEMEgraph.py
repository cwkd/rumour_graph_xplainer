# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import transformers
from joblib import Parallel, delayed
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
import torch
import json
import gc

cwd = os.getcwd()


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


def constructDataMatrix(tree, tokeniser, model, device, verbose=False):
    tweetids: [str] = list(filter(lambda x: x.isnumeric(), tree.keys()))
    root_tweetid: int = tree['root_tweetid']
    row, col = [], []  # sparse matrix representation of adjacency matrix
    idx_counter = 1  # Counter to track current node index to be assigned
    id2index: {str: int} = {f'{root_tweetid}': 0}  # Dictionary to for fast node index lookup from tweet ID
    root_index: int = 0  # Root tweet node index; set to 0 by default
    texts: [str] = [tree[f'{root_tweetid}']['text']]  # First row of texts
    label: int = tree['label']
    no_parent_tweetids, missing_parent_tweetids = set(), set()
    temp_graph: {str: [str]} = {k: [] for k in tweetids}
    # Assign children to parents
    for tweetid in tweetids:
        tweetid: str
        parent_tweetid: int = tree[tweetid]['parent_tweetid']
        if parent_tweetid is not None:
            # check = tweetid_check.get(f'{parent_tweetid}', False)
            # print(parent_tweetid, tweetid, check)
            # if check:
            try:
                temp_graph[f'{parent_tweetid}'].append(tweetid)
            except:
                temp_graph[f'{parent_tweetid}'] = [tweetid]
    # Add first level of reactions
    tweetid_check: {str: bool} = {child_tweetid: True for child_tweetid in temp_graph[f'{root_tweetid}']}
    for child_tweetid in tweetid_check.keys():
        texts.append(tree[tweetid]['text'])
        row.append(root_index)
        col.append(idx_counter)
        id2index[child_tweetid] = idx_counter
        idx_counter += 1
    tweetid_check[f'{root_tweetid}'] = True
    # Progressively construct tree
    for tweetid in tweetids:
        parent_tweetid: int = tree[tweetid]['parent_tweetid']
        if parent_tweetid is None:  # Skip tweets without parent
            if tweetid != f'{root_tweetid}':
                no_parent_tweetids.add(tweetid)
            continue
        if tweetid != f'{root_tweetid}':  # Check that tweet ID is not root tweet ID
            if tweetid_check.get(f'{parent_tweetid}', False):  # Check that tweet parent is in current tree
                for child_tweetid in temp_graph[tweetid]:
                    assert type(child_tweetid) is str
                    try:
                        id2index[child_tweetid]
                    except:
                        texts.append(tree[child_tweetid]['text'])
                        row.append(id2index[tweetid])
                        col.append(idx_counter)
                        tweetid_check[child_tweetid] = True  # Add child tweets to current tree
                        id2index[child_tweetid] = idx_counter
                        idx_counter += 1
            else:
                missing_parent_tweetids.add(tweetid)
                # print(f'Node Error: {parent_tweetid} not in current tree {root_tweetid}')

    # Log for sanity checking
    if verbose:
        if len(row) != 0:
            check = False
            if max(row) < len(texts) and max(col) < len(texts):
                check = True
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'Max Origin Index: {max(row)}\tMax Dest Index: {max(col)}\tMax Index < Num Texts: {check}')
            print('Parents not in tree: ', missing_parent_tweetids)
            print('No parent IDs: ', no_parent_tweetids)
        else:
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'No Reactions')
    try:
        assert idx_counter == len(texts)
        assert idx_counter <= len(tweetids)
    except:
        pass
    processing_metadata = {'num_tweetids': len(tweetids),
                           'num_embeddings': len(texts),
                           'origin_index_max': max(row) if len(row) != 0 else None,
                           'dest_index_max': max(col) if len(col) != 0 else None,
                           'num_missing_parents': len(missing_parent_tweetids),
                           'num_no_parents': len(no_parent_tweetids)}
    # Batch encode texts with BERT
    try:
        with torch.no_grad():
            encoded_texts: transformers.BatchEncoding = tokeniser(texts,
                                      padding='max_length',
                                      max_length=256,
                                      truncation=True,
                                      return_tensors='pt')
            tokens = []
            for text in texts:
                tokens.append(tokeniser.tokenize(text))
            embeddings = model.embeddings(encoded_texts['input_ids'].to(device)).cpu().detach()
            cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
            root_feat = embeddings[root_index].reshape(-1, 256 * 768).cpu().detach().numpy()
            # x_word = torch.cat([embeddings[:root_index], embeddings[root_index+1:]], dim=0).reshape(-1, 256*768).cpu().detach().numpy()
            x_word = embeddings.reshape(-1, 256 * 768).cpu().detach().numpy()
            return x_word, cls, tokens, [row, col], root_feat, root_index, label, tweetids, processing_metadata
    except:
        # print(root_tweetid, tweetids, texts)
        raise Exception


def getRawData(eventname, tree_id, verbose=False):
    event_dir_path = os.path.join(cwd, 'data', 'PHEME')
    event_json_path = os.path.join(event_dir_path, f'{eventname}.json')
    with open(event_json_path, 'r') as event_json_file:
        event = json.load(event_json_file)
    # event_tweets = list(event.keys())
    # print(tree_id)
    tree = event[tree_id]
    # for event_json in filter(lambda x: x.find('.json') != -1, os.listdir(event_dir_path)):
    #     event_json_path = os.path.join(event_dir_path, event_json)
    #     with open(event_json_path, 'r') as event_json_file:
    #         event = json.load(event_json_file)
    #     # for root_tweetid in event.keys():
    #     #     tree = event[root_tweetid]
    #     print('loading dataset')
    #     eventname = event_json.split('.')[0]
    #     event_tweets = list(event.keys())
    #     print(event_tweets)
    #     raise Exception
    tweetids: [str] = list(filter(lambda x: x.isnumeric(), tree.keys()))
    root_tweetid: int = tree['root_tweetid']
    row, col = [], []  # sparse matrix representation of adjacency matrix
    idx_counter = 1  # Counter to track current node index to be assigned
    id2index: {str: int} = {f'{root_tweetid}': 0}  # Dictionary to for fast node index lookup from tweet ID
    root_index: int = 0  # Root tweet node index; set to 0 by default
    texts: [str] = [tree[f'{root_tweetid}']['text']]  # First row of texts
    label: int = tree['label']
    no_parent_tweetids, missing_parent_tweetids = set(), set()
    temp_graph: {str: [str]} = {k: [] for k in tweetids}
    # Assign children to parents
    for tweetid in tweetids:
        tweetid: str
        parent_tweetid: int = tree[tweetid]['parent_tweetid']
        if parent_tweetid is not None:
            # check = tweetid_check.get(f'{parent_tweetid}', False)
            # print(parent_tweetid, tweetid, check)
            # if check:
            try:
                temp_graph[f'{parent_tweetid}'].append(tweetid)
            except:
                temp_graph[f'{parent_tweetid}'] = [tweetid]
    # Add first level of reactions
    tweetid_check: {str: bool} = {child_tweetid: True for child_tweetid in temp_graph[f'{root_tweetid}']}
    for child_tweetid in tweetid_check.keys():
        texts.append(tree[tweetid]['text'])
        row.append(root_index)
        col.append(idx_counter)
        id2index[child_tweetid] = idx_counter
        idx_counter += 1
    tweetid_check[f'{root_tweetid}'] = True
    # Progressively construct tree
    for tweetid in tweetids:
        parent_tweetid: int = tree[tweetid]['parent_tweetid']
        if parent_tweetid is None:  # Skip tweets without parent
            if tweetid != f'{root_tweetid}':
                no_parent_tweetids.add(tweetid)
            continue
        if tweetid != f'{root_tweetid}':  # Check that tweet ID is not root tweet ID
            if tweetid_check.get(f'{parent_tweetid}', False):  # Check that tweet parent is in current tree
                for child_tweetid in temp_graph[tweetid]:
                    assert type(child_tweetid) is str
                    try:
                        id2index[child_tweetid]
                    except:
                        texts.append(tree[child_tweetid]['text'])
                        row.append(id2index[tweetid])
                        col.append(idx_counter)
                        tweetid_check[child_tweetid] = True  # Add child tweets to current tree
                        id2index[child_tweetid] = idx_counter
                        idx_counter += 1
            else:
                missing_parent_tweetids.add(tweetid)
                # print(f'Node Error: {parent_tweetid} not in current tree {root_tweetid}')

    # Log for sanity checking
    if verbose:
        if len(row) != 0:
            check = False
            if max(row) < len(texts) and max(col) < len(texts):
                check = True
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'Max Origin Index: {max(row)}\tMax Dest Index: {max(col)}\tMax Index < Num Texts: {check}')
            print('Parents not in tree: ', missing_parent_tweetids)
            print('No parent IDs: ', no_parent_tweetids)
        else:
            print(f'Sanity check: Root ID: {root_tweetid}\tNum Tweet IDs: {len(tweetids)}\tNum Texts: {len(texts)}\t'
                  f'No Reactions')
    try:
        assert idx_counter == len(texts)
        assert idx_counter <= len(tweetids)
    except:
        pass
    processing_metadata = {'num_tweetids': len(tweetids),
                           'num_embeddings': len(texts),
                           'origin_index_max': max(row) if len(row) != 0 else None,
                           'dest_index_max': max(col) if len(col) != 0 else None,
                           'num_missing_parents': len(missing_parent_tweetids),
                           'num_no_parents': len(no_parent_tweetids)}
    return texts, processing_metadata


def saveTree(tree, tokeniser, model, device, processing_metadata_dict, eventname):
    data_matrix = constructDataMatrix(tree, tokeniser, model, device)
    x_word, cls, tokens, edgeindex, root_feat, root_index, label, tweetids, processing_metadata = data_matrix
    root_tweetid = f'{tree["root_tweetid"]}'
    try:
        processing_metadata_dict[eventname][root_tweetid] = processing_metadata
    except:
        processing_metadata_dict[eventname] = {root_tweetid: processing_metadata}
    if label is None:
        print(f'{root_tweetid}: Label is None')
        return processing_metadata
    tokens = np.array(tokens)
    edgeindex = np.array(edgeindex)
    root_index = np.array(root_index)
    label = np.array(label)
    tweetids = np.array(tweetids)
    try:
        np.savez(os.path.join(cwd, 'data', 'PHEMEgraph', f'{root_tweetid}.npz'),
                 x=x_word,
                 cls=cls,
                 root=root_feat,
                 edgeindex=edgeindex,
                 rootindex=root_index,
                 y=label,
                 tokens=tokens,
                 tweetids=tweetids)
        del x_word, cls, tokens, edgeindex, root_feat, root_index, label, tweetids, processing_metadata, data_matrix, model, tokeniser
        gc.collect()
        torch.cuda.empty_cache()
    except:
        try:
            os.makedirs(os.path.join(cwd, 'data', 'PHEMEgraph'))
            print(f"Created graph directory: {os.path.join(cwd, 'data', 'PHEMEgraph')}")
        except:
            pass
    # else:
    #     print(f'Root: {root_tweetid}\t\tTweetid: {tweetids.shape}'
    #           f'\t\tEmbeds: {x_word.shape}\t\tCLS: {cls.shape}')
    # return processing_metadata


def main():
    event_dir_path = os.path.join(cwd, 'data', 'PHEME')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokeniser = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
    # tokeniser = None
    # model = None
    print('loading trees')
    processing_metadata_dict = {}
    for event_json in filter(lambda x: x.find('.json') != -1, os.listdir(event_dir_path)):
        event_json_path = os.path.join(event_dir_path, event_json)
        with open(event_json_path, 'r') as event_json_file:
            event = json.load(event_json_file)
        # for root_tweetid in event.keys():
        #     tree = event[root_tweetid]
        print('loading dataset')
        eventname = event_json.split('.')[0]
        event_tweets = list(event.keys())
        # for root_tweetid in tqdm(event_tweets):
        Parallel(n_jobs=30, backend='threading')(delayed(saveTree)(
            event[root_tweetid], tokeniser, model, device,
            processing_metadata_dict, eventname) for root_tweetid in tqdm(event_tweets))
    summary = ''
    for event, event_tweet_list in processing_metadata_dict.items():
        event_num_trees = 0
        event_num_tweetids = 0
        event_num_embeddings = 0
        event_num_missing_parents = 0
        event_num_no_parents = 0
        for _, tree_processing_metadata in event_tweet_list.items():
            event_num_trees += 1
            event_num_tweetids += tree_processing_metadata['num_tweetids']
            event_num_embeddings += tree_processing_metadata['num_embeddings']
            event_num_missing_parents += tree_processing_metadata['num_missing_parents']
            event_num_no_parents += tree_processing_metadata['num_no_parents']
        summary += f'Event Name: {event}\n' \
                   f'Num Trees: {event_num_trees}|\tNum Tweets: {event_num_tweetids}|\t' \
                   f'Num Embeddings: {event_num_embeddings}|\n' \
                   f'Num Tweets with Parents not in Tree: {event_num_missing_parents}|\t' \
                   f'Num Tweets which are not Roots with no Parents: {event_num_no_parents}\n'
    print(summary)
    return processing_metadata_dict


if __name__ == '__main__':
    main()