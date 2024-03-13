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


def constructDataMatrix(thread, tokeniser, model, device, label, max_tree_size, verbose=False):
    # msg_ids: [str] = list(filter(lambda x: x.isnumeric(), thread_json.keys()))
    msg_ids: [str] = list(map(lambda x: x['mid'], thread))
    root_msg_id: str = thread[0]['mid']
    print(root_msg_id, len(thread))
    row, col = [], []  # sparse matrix representation of adjacency matrix
    idx_counter = 1  # Counter to track current node index to be assigned
    id2index: {str: int} = {f'{root_msg_id}': 0}  # Dictionary for fast node index lookup from message ID
    root_index: int = 0  # Root tweet node index; set to 0 by default
    # texts: [str] = [thread_json[f'{root_msg_id}']['original_text']]  # First row of texts
    texts: [str] = [thread[0]['original_text']]  # First row of texts
    texts_dict: {str: str} = {}
    # label: int = thread_json['label']
    if type(label) is str:
        label = int(label)
    no_parent_tweetids, missing_parent_tweetids = set(), set()
    temp_graph: {str: [str]} = {k: [] for k in msg_ids}
    # print(len(msg_ids), msg_ids)
    # Assign children to parents
    for i, msg_id in enumerate(msg_ids):
        msg_id: str
        parent_msg_id: str = thread[i]['parent']
        texts_dict[msg_id] = thread[i]['original_text']
        if parent_msg_id is not None:
            # check = msg_id_check.get(f'{parent_msg_id}', False)
            # print(parent_msg_id, msg_id, check)
            # if check:
            try:
                temp_graph[f'{parent_msg_id}'].append(msg_id)
            except:
                temp_graph[f'{parent_msg_id}'] = [msg_id]
    # print(temp_graph)
    # Add first level of reactions
    msg_id_check: {str: bool} = {child_msg_id: True for child_msg_id in temp_graph[f'{root_msg_id}']}
    for child_msg_id in msg_id_check.keys():
        texts.append(texts_dict[msg_id])
        row.append(root_index)
        col.append(idx_counter)
        id2index[child_msg_id] = idx_counter
        idx_counter += 1
    msg_id_check[f'{root_msg_id}'] = True
    # Progressively construct thread_json
    for i, msg_id in enumerate(msg_ids):
        parent_msg_id: int = thread[i]['parent']
        if parent_msg_id is None:  # Skip tweets without parent
            if msg_id != f'{root_msg_id}':
                no_parent_tweetids.add(msg_id)
            continue
        if msg_id != f'{root_msg_id}':  # Check that tweet ID is not root tweet ID
            if msg_id_check.get(f'{parent_msg_id}', False):  # Check that tweet parent is in current thread_json
                for child_msg_id in temp_graph[msg_id]:
                    assert type(child_msg_id) is str
                    try:
                        id2index[child_msg_id]
                    except:
                        texts.append(thread[i]['original_text'])
                        row.append(id2index[msg_id])
                        col.append(idx_counter)
                        msg_id_check[child_msg_id] = True  # Add child tweets to current thread_json
                        id2index[child_msg_id] = idx_counter
                        idx_counter += 1
            else:
                missing_parent_tweetids.add(msg_id)
                # print(f'Node Error: {parent_msg_id} not in current thread_json {root_msg_id}')
    # Log for sanity checking
    if verbose:
        if len(row) != 0:
            check = False
            if max(row) < len(texts) and max(col) < len(texts):
                check = True
            print(f'Sanity check: Root ID: {root_msg_id}\tNum Tweet IDs: {len(msg_ids)}\tNum Texts: {len(texts)}\t'
                  f'Max Origin Index: {max(row)}\tMax Dest Index: {max(col)}\tMax Index < Num Texts: {check}')
            print('Parents not in thread_json: ', missing_parent_tweetids)
            print('No parent IDs: ', no_parent_tweetids)
        else:
            print(f'Sanity check: Root ID: {root_msg_id}\tNum Tweet IDs: {len(msg_ids)}\tNum Texts: {len(texts)}\t'
                  f'No Reactions')
    try:
        assert idx_counter == len(texts)
        assert idx_counter <= len(msg_ids)
    except:
        pass
    processing_metadata = {'num_tweetids': len(msg_ids),
                           'num_embeddings': len(texts),
                           'origin_index_max': max(row) if len(row) != 0 else None,
                           'dest_index_max': max(col) if len(col) != 0 else None,
                           'num_missing_parents': len(missing_parent_tweetids),
                           'num_no_parents': len(no_parent_tweetids)}
    # Batch encode texts with BERT
    if len(thread) >= 50:
        minibatch_size = 50
    else:
        minibatch_size = None
    try:
        if minibatch_size is not None:
            embeddings, cls = None, None
            if len(thread) >= max_tree_size:
                num_minibatches = max_tree_size // minibatch_size   # 10000 / 100 = 100
                print(num_minibatches)
            else:
                num_minibatches = (len(thread) // minibatch_size) + 1  # 201 / 100 = 2 => 3
            # quotient = len(thread) % minibatch_size  # 201 % 100 = 1
            for minibatch_num in range(num_minibatches):
                if minibatch_num != num_minibatches - 1:
                    temp_texts = texts[minibatch_num * minibatch_size: (minibatch_num + 1) * minibatch_size]
                else:
                    temp_texts = texts[minibatch_num * minibatch_size:]
                if len(temp_texts) != 0:
                    with torch.no_grad():
                        encoded_texts: transformers.BatchEncoding = tokeniser(temp_texts,
                                                                              padding='max_length',
                                                                              max_length=256,
                                                                              truncation=True,
                                                                              return_tensors='pt')
                        tokens = []
                        for text in temp_texts:
                            tokens.append(tokeniser.tokenize(text))
                        # for text in encoded_texts['input_ids']
                        # for text in texts:
                        #     tokens.append(tokeniser(text,
                        #                             padding='max_length',
                        #                             max_length=256,
                        #                             truncation=True,
                        #                             return_tensors='pt'))
                        # temp_embeddings = model.embeddings(encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
                        # attention_mask = encoded_texts['attention_mask']
                        temp_cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
                        # root_feat = embeddings[root_index].reshape(-1, 256 * 768).cpu().detach().numpy()
                        # x_word = torch.cat([embeddings[:root_index], embeddings[root_index+1:]],
                        #                    dim=0).reshape(-1, 256*768).cpu().detach().numpy()
                        del encoded_texts
                        gc.collect()
                        torch.cuda.empty_cache()
                        if embeddings is not None:
                            # print(embeddings.shape, temp_embeddings.shape, cls.shape, temp_cls.shape)
                            # embeddings = np.concatenate((embeddings, temp_embeddings), axis=0)
                            cls = np.concatenate((cls, temp_cls), axis=0)
                        else:
                            # embeddings = temp_embeddings
                            cls = temp_cls
            # print(embeddings.shape, cls.shape)
        else:
            with torch.no_grad():
                encoded_texts: transformers.BatchEncoding = tokeniser(texts,
                                          padding='max_length',
                                          max_length=256,
                                          truncation=True,
                                          return_tensors='pt')
                tokens = []
                for text in texts:
                    tokens.append(tokeniser.tokenize(text))
                # for text in encoded_texts['input_ids']
                # for text in texts:
                #     tokens.append(tokeniser(text,
                #                             padding='max_length',
                #                             max_length=256,
                #                             truncation=True,
                #                             return_tensors='pt'))
                embeddings = model.embeddings(encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
                # attention_mask = encoded_texts['attention_mask']
                cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
                # root_feat = embeddings[root_index].reshape(-1, 256 * 768).cpu().detach().numpy()
                # x_word = torch.cat([embeddings[:root_index], embeddings[root_index+1:]],
                #                    dim=0).reshape(-1, 256*768).cpu().detach().numpy()
                del encoded_texts
                gc.collect()
                torch.cuda.empty_cache()
        x_word = None
        # x_word = embeddings.reshape(-1, 256 * 768)
        # root_feat = x_word[0]
        root_feat = None
        return x_word, cls, tokens, [row, col], root_feat, root_index, label, msg_ids, processing_metadata
    except:
        # print(root_msg_id, msg_ids, texts)
        raise Exception


def saveTree(thread_json, threads_dir_path, tokeniser, model, device, processing_metadata_dict, labels_dict,
             max_tree_size):
    eid = os.path.basename(thread_json).split('.')[0]
    label = int(labels_dict[eid])
    thread_json_path = os.path.join(threads_dir_path, thread_json)
    with open(thread_json_path, 'r', encoding='utf-8', errors='replace') as thread_json_file:
        thread = json.load(thread_json_file)
    data_matrix = constructDataMatrix(thread, tokeniser, model, device, label, max_tree_size)
    x_word, cls, tokens, edgeindex, root_feat, root_index, label, tweetids, processing_metadata = data_matrix
    root_tweetid = f'{eid}'
    processing_metadata_dict[root_tweetid] = processing_metadata
    if label is None:
        print(f'{root_tweetid}: Label is None')
        return processing_metadata
    tokens = np.array(tokens)
    edgeindex = np.array(edgeindex)
    root_index = np.array(root_index)
    label = np.array(label)
    tweetids = np.array(tweetids)
    if not os.path.exists(os.path.join(cwd, 'data', 'MaWeibograph')):
        os.makedirs(os.path.join(cwd, 'data', 'MaWeibograph'))
        print(f"Created graph directory: {os.path.join(cwd, 'data', 'MaWeibograph')}")
    try:
        np.savez(os.path.join(cwd, 'data', 'MaWeibograph', f'{root_tweetid}.npz'),
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
        print(f'Thread {eid}: failed to process')
        # try:
        #     os.makedirs(os.path.join(cwd, 'data', 'MaWeibograph'))
        #     print(f"Created graph directory: {os.path.join(cwd, 'data', 'MaWeibograph')}")
        # except:
        #     pass
    # raise Exception
    # else:
    #     print(f'Root: {root_tweetid}\t\tTweetid: {tweetids.shape}'
    #           f'\t\tEmbeds: {x_word.shape}\t\tCLS: {cls.shape}')
    # return processing_metadata


def main():
    threads_dir_path = os.path.join(cwd, 'data', 'MaWeibo', 'Threads')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokeniser = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
    # tokeniser = None
    # model = None
    print('loading labels')
    labels_path = os.path.join(cwd, 'data', 'MaWeibo', 'Weibo.txt')
    # labels_output_path = os.path.join(cwd, 'data', 'MaWeibo', 'MaWeibolabels.txt')
    labels = []
    labels_dict = {}
    with open(labels_path, 'r') as labels_file:
        lines = labels_file.readlines()
    for line in lines:
        line = line.split(' ')[0]
        line = line.split('\t')
        eid = line[0].split(':')[1]
        label = line[1].split(':')[1]
        labels.append((eid, label))
        labels_dict[eid] = label
    # print(len(labels))
    print('loading trees')
    processing_metadata_dict = {}
    max_tree_size = 500
    for thread_json in tqdm(filter(lambda x: x.find('.json') != -1,
                                   os.listdir(threads_dir_path))):
        saveTree(thread_json, threads_dir_path, tokeniser, model, device, processing_metadata_dict, labels_dict,
                 max_tree_size)
    # Parallel(n_jobs=5, backend='threading')(delayed(saveTree)(
    #     thread_json, threads_dir_path, tokeniser, model, device,
    #     processing_metadata_dict, labels_dict) for thread_json in tqdm(filter(lambda x: x.find('.json') != -1,
    #                                                                           os.listdir(threads_dir_path))))

    summary = ''
    for thread, event_tweet_list in processing_metadata_dict.items():
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
        summary += f'Event Name: {thread}\n' \
                   f'Num Trees: {event_num_trees}|\tNum Tweets: {event_num_tweetids}|\t' \
                   f'Num Embeddings: {event_num_embeddings}|\n' \
                   f'Num Tweets with Parents not in Tree: {event_num_missing_parents}|\t' \
                   f'Num Tweets which are not Roots with no Parents: {event_num_no_parents}\n'
    print(summary)
    return processing_metadata_dict


if __name__ == '__main__':
    main()
