import random
from random import shuffle
import os
import numpy as np
from sklearn.model_selection import KFold
random.seed(42)

cwd = os.getcwd()


def load5foldDataStratified(obj):
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
        kf = KFold(5, random_state=0, shuffle=True)
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
    elif obj == 'MaWeibo':
        graph_data_dir_path = os.path.join(cwd, 'data', f'{obj}graph')
        nr, tr = [], []
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
        nr = np.asarray(nr)
        tr = np.asarray(tr)
        folds = []
        kf = KFold(5, random_state=0, shuffle=True)
        for train_index, test_index in kf.split(nr):
            train, test = nr[train_index], nr[test_index]
            folds.append([train, test])
        for fold_num, (train_index, test_index) in enumerate(kf.split(tr)):
            train, test = tr[train_index], tr[test_index]
            folds[fold_num].append(train)
            folds[fold_num].append(test)
        assert len(folds) == 5
        assert len(folds[0]) == 4
        return folds


def load5foldData(obj):
    if 'Twitter' in obj:
        labelPath = os.path.join(cwd, "data", "Twitter1516_label_All.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("loading tree label")
        NR, F, T, U = [], [], [], []
        l1 = l2 = l3 = l4 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            if labelDic.get(eid, None) is not None:
                # print(f'Duplicate ID: {eid}')
                continue
            labelDic[eid] = label.lower()
            if label in labelset_nonR:
                NR.append(eid)
                l1 += 1
            if labelDic[eid] in labelset_f:
                F.append(eid)
                l2 += 1
            if labelDic[eid] in labelset_t:
                T.append(eid)
                l3 += 1
            if labelDic[eid] in labelset_u:
                U.append(eid)
                l4 += 1
        print(len(labelDic))
        print(l1, l2, l3, l4)
        random.Random(0).shuffle(NR)
        random.Random(0).shuffle(F)
        random.Random(0).shuffle(T)
        random.Random(0).shuffle(U)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)

        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng3])
        fold0_x_test.extend(U[0:leng4])
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])
        fold1_x_train.extend(NR[0:leng1])
        fold1_x_train.extend(NR[leng1 * 2:])
        fold1_x_train.extend(F[0:leng2])
        fold1_x_train.extend(F[leng2 * 2:])
        fold1_x_train.extend(T[0:leng3])
        fold1_x_train.extend(T[leng3 * 2:])
        fold1_x_train.extend(U[0:leng4])
        fold1_x_train.extend(U[leng4 * 2:])
        fold1_x_test.extend(NR[leng1:leng1*2])
        fold1_x_test.extend(F[leng2:leng2*2])
        fold1_x_test.extend(T[leng3:leng3*2])
        fold1_x_test.extend(U[leng4:leng4*2])
        fold2_x_train.extend(NR[0:leng1*2])
        fold2_x_train.extend(NR[leng1*3:])
        fold2_x_train.extend(F[0:leng2*2])
        fold2_x_train.extend(F[leng2*3:])
        fold2_x_train.extend(T[0:leng3*2])
        fold2_x_train.extend(T[leng3*3:])
        fold2_x_train.extend(U[0:leng4*2])
        fold2_x_train.extend(U[leng4*3:])
        fold2_x_test.extend(NR[leng1*2:leng1*3])
        fold2_x_test.extend(F[leng2*2:leng2*3])
        fold2_x_test.extend(T[leng3*2:leng3*3])
        fold2_x_test.extend(U[leng4*2:leng4*3])
        fold3_x_train.extend(NR[0:leng1*3])
        fold3_x_train.extend(NR[leng1*4:])
        fold3_x_train.extend(F[0:leng2*3])
        fold3_x_train.extend(F[leng2*4:])
        fold3_x_train.extend(T[0:leng3*3])
        fold3_x_train.extend(T[leng3*4:])
        fold3_x_train.extend(U[0:leng4*3])
        fold3_x_train.extend(U[leng4*4:])
        fold3_x_test.extend(NR[leng1*3:leng1*4])
        fold3_x_test.extend(F[leng2*3:leng2*4])
        fold3_x_test.extend(T[leng3*3:leng3*4])
        fold3_x_test.extend(U[leng4*3:leng4*4])
        fold4_x_train.extend(NR[0:leng1*4])
        fold4_x_train.extend(NR[leng1*5:])
        fold4_x_train.extend(F[0:leng2*4])
        fold4_x_train.extend(F[leng2*5:])
        fold4_x_train.extend(T[0:leng3*4])
        fold4_x_train.extend(T[leng3*5:])
        fold4_x_train.extend(U[0:leng4*4])
        fold4_x_train.extend(U[leng4*5:])
        fold4_x_test.extend(NR[leng1*4:leng1*5])
        fold4_x_test.extend(F[leng2*4:leng2*5])
        fold4_x_test.extend(T[leng3*4:leng3*5])
        fold4_x_test.extend(U[leng4*4:leng4*5])

    if obj == "Weibo":
        labelPath = os.path.join(cwd,"data/Weibo/weibo_id_label.txt")
        print("loading weibo label:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid,label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)
            if labelDic[eid]==0:
                F.append(eid)
                l1 += 1
            if labelDic[eid]==1:
                T.append(eid)
                l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.shuffle(F)
        random.shuffle(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])

    fold0_test = list(fold0_x_test)
    random.Random(0).shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    random.Random(0).shuffle(fold0_train)
    fold1_test = list(fold1_x_test)
    random.Random(0).shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    random.Random(0).shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    random.Random(0).shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    random.Random(0).shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    random.Random(0).shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    random.Random(0).shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    random.Random(0).shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    random.Random(0).shuffle(fold4_train)

    return list(fold0_test),list(fold0_train),\
           list(fold1_test),list(fold1_train),\
           list(fold2_test),list(fold2_train),\
           list(fold3_test),list(fold3_train),\
           list(fold4_test), list(fold4_train)
