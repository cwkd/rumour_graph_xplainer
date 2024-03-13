import random
from random import shuffle
import os
import json
import copy
import numpy as np
from sklearn.model_selection import KFold

cwd = os.getcwd()


def load9foldData(obj):
    # labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_All.txt")
    graph_data_dir_path = os.path.join(cwd, 'data', f'{obj}graph')
    graph_data_check = {tree_id.split('.')[0]: True for tree_id in os.listdir(graph_data_dir_path)}
    data_dir_path = os.path.join(cwd, 'data', obj)
    event_jsons = sorted(list(filter(lambda x: x.find('.json') != -1, os.listdir(data_dir_path))))
    # print(event_jsons)
    # labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'nonrumor'], ['false'], ['true'], ['unverified']
    print("loading tree label")
    # NR,F,T,U = [],[],[],[]
    # l1=l2=l3=l4=0
    # labelDic = {}
    train_folds, test_folds = [], []
    for fold_num, event_json in enumerate(event_jsons):
        event_jsons_copy = copy.copy(event_jsons)
        event_jsons_copy.remove(event_json)
        train_event_ids = []
        train_event_labels = []
        # print(event_json, event_jsons_copy)
        for current_event in event_jsons_copy:
            event_json_path = os.path.join(data_dir_path, current_event)
            with open(event_json_path, 'r') as event:
                tweets = json.load(event)
            # print(list(filter(lambda x: graph_data_check.get(x, False), tweets.keys())))
            train_event_ids += list(filter(lambda x: graph_data_check.get(x, False), tweets.keys()))
            for tweetid in tweets.keys():
                train_event_labels.append(tweets[tweetid]['label'])
        train_folds.append((train_event_ids, train_event_labels))
        event_json_path = os.path.join(data_dir_path, event_json)
        test_event_labels = []
        with open(event_json_path, 'r') as event:
            tweets = json.load(event)
            # print(list(filter(lambda x: graph_data_check.get(x, False), tweets.keys())))
            test_event_ids = list(filter(lambda x: graph_data_check.get(x, False), tweets.keys()))
            for tweetid in tweets.keys():
                test_event_labels.append(tweets[tweetid]['label'])
        test_folds.append((test_event_ids, test_event_labels))
    return list(zip(train_folds, test_folds))


def load9foldDataStratified(obj):
    if obj == 'PHEME':
        graph_data_dir_path = os.path.join(cwd, 'data', f'{obj}graph')
        # graph_data_check = {tree_id.split('.')[0]: True for tree_id in os.listdir(graph_data_dir_path)}
        # data_dir_path = os.path.join(cwd, 'data', obj)
        # event_jsons = sorted(list(filter(lambda x: x.find('.json') != -1, os.listdir(data_dir_path))))
        nr, tr, fr, uv = [], [], [], []
        eids = []
        for data_file in os.listdir(graph_data_dir_path):
            data = np.load(os.path.join(graph_data_dir_path, data_file), allow_pickle=True)
            eid = data_file.split('.')[0]
            label = int(data['y'])
            eids.append(eid)
            if label == 0:
                nr.append(eid)
            elif label == 1:
                tr.append(eid)
            elif label == 2:
                fr.append(eid)
            elif label == 3:
                uv.append(eid)
        nr = np.asarray(nr)
        tr = np.asarray(tr)
        fr = np.asarray(fr)
        uv = np.asarray(uv)
        folds = []
        kf = KFold(9, random_state=0, shuffle=True)
        for train_index, test_index in kf.split(nr):
            train, test = nr[train_index], nr[test_index]
            folds.append([train, test])
        for fold_num, (train_index, test_index) in enumerate(kf.split(tr)):
            train, test = tr[train_index], tr[test_index]
            folds[fold_num].append(train)
            folds[fold_num].append(test)
        for fold_num, (train_index, test_index) in enumerate(kf.split(fr)):
            train, test = fr[train_index], fr[test_index]
            folds[fold_num].append(train)
            folds[fold_num].append(test)
        for fold_num, (train_index, test_index) in enumerate(kf.split(uv)):
            train, test = uv[train_index], uv[test_index]
            folds[fold_num].append(train)
            folds[fold_num].append(test)
        assert len(folds) == 5
        assert len(folds[0]) == 8
        return folds
