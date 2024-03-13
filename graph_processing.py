import os
from pyvis.network import Network
import networkx as nx
from datetime import datetime
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt
import json
from Process.rand5fold import load5foldDataStratified, load5foldData
from Process.process import loadBiData, loadTree
import numpy as np
from torch_geometric.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CENTRALITY_DIR = os.path.join(DATA_DIR, 'centrality')
FOLD_2_EVENTNAME = {0: 'charliehebdo',
                    1: 'ebola',
                    2: 'ferguson',
                    3: 'germanwings',
                    4: 'gurlitt',
                    5: 'ottawashooting',
                    6: 'prince',
                    7: 'putinmissing',
                    8: 'sydneysiege'}
MONTH_TO_MONTHNUM = {'jan': '01',
                     'feb': '02',
                     'mar': '03',
                     'apr': '04',
                     'may': '05',
                     'jun': '06',
                     'jul': '07',
                     'aug': '08',
                     'sep': '09',
                     'oct': '10',
                     'nov': '11',
                     'dec': '12'
                     }


def parse_tweet_time(tweet_time):
    _, month, day, time, _, year = tweet_time.split(' ')
    month_num = MONTH_TO_MONTHNUM[month.lower()]
    date_time_string = f'{year}-{month_num}-{day} {time}'
    date_time_obj = datetime.strptime(date_time_string, '%Y-%m-%d %H:%M:%S')
    return date_time_obj.timestamp()


def format_tweet_text(text):
    words = text.split(' ')
    output_text = ''
    word_num = 5
    quotient = len(words) // word_num
    # remainder = len(words) % word_num
    for segment in range(len(words) // word_num):
        segment_start = segment * 5
        segment_end = (segment + 1) * 5
        output_text += ' '.join(words[segment_start:segment_end]) + '<br>'
    else:
        if quotient == 0:
            return text
        else:
            output_text += ' '.join(words[segment_end:])
    return output_text


def plot_tweet_graphs(tweet_meta_info, tree_info, dataset='PHEME', event_name='charliehebdo'):
    with open(os.path.join(DATA_DIR, dataset, f'{event_name}.json'), 'r') as jsonfile:
        data = json.load(jsonfile)
    tree_ids = data.keys()
    # td_graph = nx.DiGraph()
    # node_num = 0
    # tweetid_to_node_num = {}
    tree_num = 0
    cmap = plt.cm.cool
    save = True
    for root_tweetid in tqdm(tree_ids):
        td_graph = nx.DiGraph()
        bu_graph = nx.DiGraph()
        node_num = 0
        tweetid_to_node_num = {}
        tree = data[root_tweetid]
        nodes = []
        td_edges = []
        bu_edges = []
        for key, tweet in tree.items():
            if not key.isnumeric():
                continue
            curr_node_num = tweetid_to_node_num.get(key, None)
            if curr_node_num is None:
                curr_node_num = node_num
                tweetid_to_node_num[key] = node_num
            try:
                tweet_time = tweet['tweet_time']
            except:
                tweet_time = None
            formatted_text = format_tweet_text(tweet['text'])
            parent_tweetid = tweet['parent_tweetid']
            userid = tweet['userid']
            if key == root_tweetid:
                node = (curr_node_num,
                        {'title': f'Tweet ID: {key}<br>User ID: {userid}<br>{formatted_text}<br>{tweet_time}',
                         'label': f'{key}',
                         'group': tree_num,
                         'color': 'black'}
                        )
            elif tree.get(f'{parent_tweetid}', None) is None:
                continue
            else:
                # parent_tweet_time = tree[f'{parent_tweetid}']['tweet_time']
                # edge_weight = parse_tweet_time(tweet_time) - parse_tweet_time(parent_tweet_time)
                colour_num = tweet_meta_info[key]['links']
                node_colour = cmap(colour_num/21)
                node = (curr_node_num,
                        {'title': f'Tweet ID: {key}<br>User ID: {userid}<br>{formatted_text}<br>{tweet_time}',
                         'label': f'{key}',
                         'group': tree_num,
                         'color': f'#{int(node_colour[0]*255):02x}'
                                       f'{int(node_colour[1]*255):02x}'
                                       f'{int(node_colour[2]*255):02x}'}
                        )
            nodes.append(node)
            node_num += 1
            if parent_tweetid is None:
                # print(key, node[1]['title'])
                continue
            parent_node_num = tweetid_to_node_num.get(f'{parent_tweetid}', None)
            # print(parent_node_num, curr_node_num, type(parent_tweetid), type(key))
            td_edge = (parent_node_num, curr_node_num)
            td_edges.append(td_edge)
            bu_edge = (curr_node_num, parent_node_num)
            bu_edges.append(bu_edge)
        else:
            # print(nodes)
            # print(edges)
            td_graph.add_nodes_from(nodes)
            td_graph.add_edges_from(td_edges)
            bu_graph.add_nodes_from(nodes)
            bu_graph.add_edges_from(bu_edges)
            tree_num += 1
            # if group_num >= 10:
            #     break
            # net = Network('600px', '1000px', directed=True)
            # net.from_nx(td_graph)
            # net.save_graph(os.path.join(EXPLAIN_DIR, dataset, f'{name}_{root_tweetid}.html'))
            # TD Graph
            centrality_json = {'td': dict(),
                               'bu': dict()}
            temp = sorted(nx.out_degree_centrality(td_graph).items(),
                          key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # centrality_json['td']['out_degree'] = temp_dict
            centrality_json['td']['out_degree'] = [list(node_list), list(score_list)]

            temp = sorted(nx.closeness_centrality(td_graph).items(),
                          key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # centrality_json['td']['closeness'] = temp_dict
            centrality_json['td']['closeness'] = [list(node_list), list(score_list)]

            temp = sorted(nx.betweenness_centrality(td_graph, normalized=True, endpoints=False).items(),
                                                    key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # centrality_json['td']['betweenness'] = temp_dict
            centrality_json['td']['betweenness'] = [list(node_list), list(score_list)]

            temp = sorted(nx.eigenvector_centrality(td_graph, 100000).items(),
                          key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # centrality_json['td']['eigenvector'] = temp_dict
            centrality_json['td']['eigenvector'] = [list(node_list), list(score_list)]

            # BU Graph
            temp = sorted(nx.out_degree_centrality(bu_graph).items(),
                          key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # centrality_json['bu']['out_degree'] = temp_dict
            centrality_json['bu']['out_degree'] = [list(node_list), list(score_list)]

            temp = sorted(nx.closeness_centrality(bu_graph).items(),
                          key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # centrality_json['bu']['closeness'] = temp_dict
            centrality_json['bu']['closeness'] = [list(node_list), list(score_list)]

            temp = sorted(nx.betweenness_centrality(bu_graph, normalized=True, endpoints=False).items(),
                          key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # centrality_json['bu']['betweenness'] = temp_dict
            centrality_json['bu']['betweenness'] = [list(node_list), list(score_list)]

            temp = sorted(nx.eigenvector_centrality(bu_graph, 100000).items(),
                          key=lambda x: x[1], reverse=True)
            node_list, score_list = (zip(*temp))
            # temp_dict = dict()
            # for node, score in zip(node_list, score_list):
            #     try:
            #         temp_dict[score].append(node)
            #     except:
            #         temp_dict[score] = [node]
            # else:
            #     for key in temp_dict.keys():
            #         temp_dict[key] = sorted(temp_dict[key])
            # # centrality_json['bu']['eigenvector'] = temp_dict
            centrality_json['bu']['eigenvector'] = [list(node_list), list(score_list)]

            # print(root_tweetid)
            # print(len(td_graph.nodes))
            # print(len(td_graph.edges))
            # print(list(nx.out_degree_centrality(td_graph).items()))
            # print(sorted(nx.out_degree_centrality(td_graph).items(), key=lambda x: x[1], reverse=True))
            # raise Exception
            if save:
                if not os.path.exists(os.path.join(CENTRALITY_DIR, dataset, event_name)):
                    os.makedirs(os.path.join(CENTRALITY_DIR, dataset, event_name))
                with open(os.path.join(CENTRALITY_DIR, dataset, event_name,
                                       f'{root_tweetid}_centrality.json'), 'w') as jsonfile:
                    json.dump(centrality_json, jsonfile, indent=1)
            tree_num += 1
        # print(tree)
        # print(nodes)
        # print(edges)
        # net = Network('500px', '500px')
        # net.add_nodes(nodes=nodes, title=title)
        # print(net.nodes)
        # net.add_edges(edges)
        # net.save_graph(f'{name}.html')
    # print(td_graph)
    # net = Network('600px', '1000px')
    # net.from_nx(td_graph)
    # net.save_graph(f'{name}_nx1.html')
    # net.show(f'{name}_nx.html')
    # return nodes, edges, td_graph


def compile_tweet_meta_info(dataset='PHEME', event_name='charliehebdo'):
    with open(os.path.join(DATA_DIR, dataset, f'{event_name}.json'), 'r') as jsonfile:
        data = json.load(jsonfile)
    tree_ids = data.keys()
    tweet_meta_info = {}
    for root_tweetid in tqdm(tree_ids):
        tree = data[root_tweetid]
        for key, tweet in tree.items():
            if not key.isnumeric():
                continue
            parent_tweetid = tweet['parent_tweetid']
            if key == root_tweetid:
                tweet_meta_info[f'{key}'] = {'tree': f'{root_tweetid}'}
                continue
            elif tree.get(f'{parent_tweetid}', None) is None:
                continue
            # print(key, parent_tweetid)
            try:
                tweet_meta_info[f'{key}']['tree'] = f'{root_tweetid}'
            except:
                tweet_meta_info[f'{key}'] = {'tree': f'{root_tweetid}'}
            # Add parent tweet id to edge source list
            try:
                tweet_meta_info[f'{key}']['in'].append(f'{parent_tweetid}')
            except:
                tweet_meta_info[f'{key}']['in'] = [f'{parent_tweetid}']
            if tweet_meta_info.get(f'{parent_tweetid}', None) is None:
                tweet_meta_info[f'{parent_tweetid}'] = {'tree': f'{root_tweetid}'}
            # Add tweet id to edge destination list in parent
            try:
                tweet_meta_info[f'{parent_tweetid}']['out'].append(f'{key}')
            except:
                tweet_meta_info[f'{parent_tweetid}']['out'] = [f'{key}']
    tweet_meta_info_dist = []
    for key, value in tqdm(tweet_meta_info.items()):
        in_edge = value.get('in', [])
        out_edge = value.get('out', [])
        links = len(in_edge) + len(out_edge)
        tree_id = value.get('tree', None)
        tweet_meta_info[key]['links'] = links
        tweet_meta_info_dist.append((key, tree_id, links, in_edge, out_edge))
    tweet_meta_info_dist = sorted(tweet_meta_info_dist, key=lambda x: x[2], reverse=True)
    return tweet_meta_info, tweet_meta_info_dist


def compile_user_info(dataset='PHEME', event_name='charliehebdo'):
    with open(os.path.join(DATA_DIR, dataset, f'{event_name}.json'), 'r') as jsonfile:
        data = json.load(jsonfile)
    tree_ids = data.keys()
    tweets_per_userid, links_per_userid = {}, {}
    threads_per_userid = {}
    for root_tweetid in tqdm(tree_ids):
        tree = data[root_tweetid]
        label = 'unverified'
        for key, tweet in tree.items():
            if key == 'label':
                if tweet == 0:
                    label = 'non-rumour'
                elif tweet == 1:
                    label = 'false-rumour'
                elif tweet == 2:
                    label = 'true-rumour'
                else:
                    label = 'unverified'
            # Skip non-numeric keys; keys which are not tweet ids
            if not key.isnumeric():
                continue
            userid = tweet['userid']
            parent_tweetid = tweet['parent_tweetid']
            # If tweet is root tweet, add only tweet count to user id
            if key == root_tweetid:
                try:
                    tweets_per_userid[f'{userid}'].append(f'{key}')
                except:
                    tweets_per_userid[f'{userid}'] = [f'{key}']
                try:
                    threads_per_userid[f'{userid}'].add(f'{root_tweetid}')
                except:
                    threads_per_userid[f'{userid}'] = {f'{root_tweetid}'}
            elif tree.get(f'{parent_tweetid}', None) is None:  # If tweet parent does not exist in tree, skip it
                continue
            else:
                # Add tweet count to user id for regular tweet
                try:
                    tweets_per_userid[f'{userid}'].append(f'{key}')
                except:
                    tweets_per_userid[f'{userid}'] = [f'{key}']
                parent_userid = tree[f'{parent_tweetid}']['userid']
                # Add link destination to user id for regular tweet
                if links_per_userid.get(f'{userid}', None) is None:
                    links_per_userid[f'{userid}'] = {'src': {}, 'dst': {}}  # Init empty dict for new userid
                try:
                    links_per_userid[f'{userid}']['src'][f'{parent_userid}'] += 1
                except:
                    links_per_userid[f'{userid}']['src'][f'{parent_userid}'] = 1
                # Add link source to user id for regular tweet
                if links_per_userid.get(f'{parent_userid}', None) is None:
                    links_per_userid[f'{parent_userid}'] = {'src': {}, 'dst': {}}  # Init empty dict for new parent userid
                try:
                    links_per_userid[f'{parent_userid}']['dst'][f'{userid}'] += 1
                except:
                    links_per_userid[f'{parent_userid}']['dst'][f'{userid}'] = 1
                try:
                    threads_per_userid[f'{userid}'].add(f'{root_tweetid}')
                except:
                    threads_per_userid[f'{userid}'] = {f'{root_tweetid}'}
    user_tweet_dist, user_link_dist = [], []
    user_thread_dist = []
    for key, tweet_list in tqdm(tweets_per_userid.items()):
        user_tweet_dist.append((key, len(tweet_list)))
    user_tweet_dist = sorted(user_tweet_dist, key=lambda x: x[1], reverse=True)
    for key, link_dict in tqdm(links_per_userid.items()):
        try:
            src_count = sum(link_dict['src'].values())
        except:
            src_count = 0
        try:
            dst_count = sum(link_dict['dst'].values())
        except:
            dst_count = 0
        user_link_dist.append((key, src_count + dst_count))
        links_per_userid[key]['src']['total'] = src_count
        links_per_userid[key]['dst']['total'] = dst_count
        pass
    user_link_dist = sorted(user_link_dist, key=lambda x: x[1], reverse=True)
    for key, value in tqdm(threads_per_userid.items()):
        threads_per_userid[key] = list(value)
        user_thread_dist.append((key, len(value)))
    user_thread_dist = sorted(user_thread_dist, key=lambda x: x[1], reverse=True)
    return tweets_per_userid, user_tweet_dist, links_per_userid, user_link_dist, threads_per_userid, user_thread_dist


def compile_tree_info(tweet_meta_info, dataset='PHEME', event_name='charliehebdo'):
    """
    Compile meta-information about tweet trees in an event
    :param event_name: name of the json file to be opened
    :return:
    """
    with open(os.path.join(DATA_DIR, dataset, f'{event_name}.json'), 'r') as jsonfile:
        data = json.load(jsonfile)
    tree_ids = data.keys()
    tree_info = {}
    for root_tweetid in tqdm(tree_ids):
        tree = data[root_tweetid]
        max_out_degree = 0
        max_links = 0
        num_tweets = 0
        for key in tree.keys():
            if key == 'label':
                if tree[key] == 0:
                    label = 'non-rumour'
                elif tree[key] == 1:
                    label = 'false-rumour'
                elif tree[key] == 2:
                    label = 'true-rumour'
                else:
                    label = 'unverified'
            tweet_info = tweet_meta_info.get(f'{key}', None)
            if tweet_info is None:
                continue
            else:
                num_tweets += 1
                out_degree = len(tweet_info.get('out', []))
                links = len(tweet_info.get('in', [])) + out_degree
                if out_degree > max_out_degree:
                    max_out_degree = out_degree
                if links > max_links:
                    max_links = links
        else:
            tree_info[f'{root_tweetid}'] = {'num_tweets': num_tweets,
                                            'max_links': max_links,
                                            'max_out_degree': max_out_degree,
                                            'label': label}
    tree_dist = []
    for tree_id, info in tqdm(tree_info.items()):
        tree_dist.append((tree_id, info['num_tweets'], info['max_links'], info['max_out_degree'], info['label']))
    tree_dist = sorted(tree_dist, key=lambda x: x[1], reverse=True)
    return tree_info, tree_dist


if __name__ == '__main__':
    split_type = '5fold'  # '5fold', '9fold'
    datasetname = 'Twitter'  # 'Twitter', 'PHEME'
    if datasetname == 'Twitter':
        for fold_num, fold in enumerate(load5foldData(datasetname)):
            if fold_num % 2 != 0:  # Training fold, skip this
                continue
            else:
                fold_num = fold_num // 2
            event_name = f'fold{fold_num}'
            treeDic = loadTree(datasetname)
            TDdroprate, BUdroprate = 0, 0
            fold_train = []
            fold_test = fold
            traindata_list, testdata_list = loadBiData(datasetname,
                                                       treeDic,
                                                       fold_train,
                                                       fold_test,
                                                       TDdroprate,
                                                       BUdroprate)
            test_loader = DataLoader(testdata_list, batch_size=1, shuffle=False, num_workers=1)
            for sample_num, (data_sample, root_tweetid) in enumerate(tqdm(test_loader)):
                root_tweetid = root_tweetid[0][0]
                td_graph = nx.DiGraph()
                bu_graph = nx.DiGraph()
                td_edges, bu_edges = [], []
                for src, dst in zip(data_sample.edge_index[0], data_sample.edge_index[1]):
                    td_edges.append((src.item(), dst.item()))
                    bu_edges.append((dst.item(), src.item()))
                td_graph.add_edges_from(td_edges)
                bu_graph.add_edges_from(bu_edges)
                if len(td_edges) != 0:
                    centrality_json = {'td': dict(),
                                       'bu': dict()}
                    temp = sorted(nx.out_degree_centrality(td_graph).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['td']['out_degree'] = [list(node_list), list(score_list)]

                    temp = sorted(nx.closeness_centrality(td_graph).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['td']['closeness'] = [list(node_list), list(score_list)]

                    temp = sorted(nx.betweenness_centrality(td_graph, normalized=True, endpoints=False).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['td']['betweenness'] = [list(node_list), list(score_list)]

                    temp = sorted(nx.eigenvector_centrality(td_graph, 100000).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['td']['eigenvector'] = [list(node_list), list(score_list)]

                    # BU Graph
                    temp = sorted(nx.out_degree_centrality(bu_graph).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['bu']['out_degree'] = [list(node_list), list(score_list)]

                    temp = sorted(nx.closeness_centrality(bu_graph).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['bu']['closeness'] = [list(node_list), list(score_list)]

                    temp = sorted(nx.betweenness_centrality(bu_graph, normalized=True, endpoints=False).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['bu']['betweenness'] = [list(node_list), list(score_list)]

                    temp = sorted(nx.eigenvector_centrality(bu_graph, 100000).items(),
                                  key=lambda x: x[1], reverse=True)
                    node_list, score_list = (zip(*temp))
                    centrality_json['bu']['eigenvector'] = [list(node_list), list(score_list)]
                    save = True
                    if save:
                        if not os.path.exists(os.path.join(CENTRALITY_DIR, datasetname, event_name)):
                            os.makedirs(os.path.join(CENTRALITY_DIR, datasetname, event_name))
                        with open(os.path.join(CENTRALITY_DIR, datasetname, event_name,
                                               f'{root_tweetid}_centrality.json'), 'w') as jsonfile:
                            json.dump(centrality_json, jsonfile, indent=1)
    else:  # PHEME
        if split_type == '5fold':
            print('Computing centrality measures, 5fold split')
            for fold_num, fold in enumerate(load5foldDataStratified(datasetname)):
                c0_train, c0_test, c1_train, c1_test, c2_train, c2_test, c3_train, c3_test = fold
                fold_train = np.concatenate((c0_train, c1_train, c2_train, c3_train)).tolist()
                fold_test = np.concatenate((c0_test, c1_test, c2_test, c3_test)).tolist()
                event_name = f'fold{fold_num}'
                treeDic = None
                TDdroprate, BUdroprate = 0, 0
                traindata_list, testdata_list = loadBiData(datasetname,
                                                           treeDic,
                                                           fold_train,
                                                           fold_test,
                                                           TDdroprate,
                                                           BUdroprate)
                test_loader = DataLoader(testdata_list, batch_size=1, shuffle=False, num_workers=5)
                for sample_num, (data_sample, root_tweetid) in enumerate(tqdm(test_loader)):
                    root_tweetid = root_tweetid[0]
                    td_graph = nx.DiGraph()
                    bu_graph = nx.DiGraph()
                    td_edges, bu_edges = [], []
                    for src, dst in zip(data_sample.edge_index[0], data_sample.edge_index[1]):
                        td_edges.append((src.item(), dst.item()))
                        bu_edges.append((dst.item(), src.item()))
                    td_graph.add_edges_from(td_edges)
                    bu_graph.add_edges_from(bu_edges)
                    if len(td_edges) != 0:
                        centrality_json = {'td': dict(),
                                           'bu': dict()}
                        temp = sorted(nx.out_degree_centrality(td_graph).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['td']['out_degree'] = [list(node_list), list(score_list)]

                        temp = sorted(nx.closeness_centrality(td_graph).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['td']['closeness'] = [list(node_list), list(score_list)]

                        temp = sorted(nx.betweenness_centrality(td_graph, normalized=True, endpoints=False).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['td']['betweenness'] = [list(node_list), list(score_list)]

                        temp = sorted(nx.eigenvector_centrality(td_graph, 100000).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['td']['eigenvector'] = [list(node_list), list(score_list)]

                        # BU Graph
                        temp = sorted(nx.out_degree_centrality(bu_graph).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['bu']['out_degree'] = [list(node_list), list(score_list)]

                        temp = sorted(nx.closeness_centrality(bu_graph).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['bu']['closeness'] = [list(node_list), list(score_list)]

                        temp = sorted(nx.betweenness_centrality(bu_graph, normalized=True, endpoints=False).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['bu']['betweenness'] = [list(node_list), list(score_list)]

                        temp = sorted(nx.eigenvector_centrality(bu_graph, 100000).items(),
                                      key=lambda x: x[1], reverse=True)
                        node_list, score_list = (zip(*temp))
                        centrality_json['bu']['eigenvector'] = [list(node_list), list(score_list)]
                        save = True
                        if save:
                            if not os.path.exists(os.path.join(CENTRALITY_DIR, datasetname, event_name)):
                                os.makedirs(os.path.join(CENTRALITY_DIR, datasetname, event_name))
                            with open(os.path.join(CENTRALITY_DIR, datasetname, event_name,
                                                   f'{root_tweetid}_centrality.json'), 'w') as jsonfile:
                                json.dump(centrality_json, jsonfile, indent=1)
            print('Completed')
        elif split_type == '9fold':
            print('Computing centrality measures, 9fold split')
            for fold_num in range(len(FOLD_2_EVENTNAME)):
                event_name = FOLD_2_EVENTNAME[fold_num]
                tweet_meta_info, tweet_meta_info_dist = compile_tweet_meta_info(dataset=datasetname,
                                                                                event_name=event_name)
                tree_info, tree_dist = compile_tree_info(tweet_meta_info,
                                                         dataset=datasetname,
                                                         event_name=event_name)
                tweets_per_userid, user_tweet_dist, links_per_userid, user_link_dist, threads_per_userid, user_thread_dist = \
                    compile_user_info(dataset=datasetname,
                                      event_name=event_name)
                plot_tweet_graphs(tweet_meta_info,
                                  tree_info,
                                  dataset=datasetname,
                                  event_name=event_name)
    print('End of programme')
    save = False
    if save:
        with open('data/tree_info.csv', 'w') as f:
            f.write('tree_id,num_tweets,max_links,max_out_degree,label')
            for (tree_id, num_tweets, max_links, max_out_degree, label) in tree_dist:
                f.write(f'\n{tree_id},{num_tweets},{max_links},{max_out_degree},{label}')

        with open('data/tree_info.json', 'w') as f:
            json.dump(tree_info, f, indent=1)

        with open('data/user_threads.json', 'w') as f:
            json.dump(threads_per_userid, f, indent=1)

        with open('data/user_tweets.json', 'w') as f:
            json.dump(tweets_per_userid, f, indent=1)

        with open('data/tweet_meta_info.csv', 'w') as f:
            f.write('tweetid,root_tweetid,link_count,in_degree,out_degree,in_edge,out_edge')
            for (tweetid, root_tweetid, link_count, in_edge, out_edge) in tweet_meta_info_dist:
                f.write(f'\n{tweetid},{root_tweetid},{link_count},{len(in_edge)},{len(out_edge)},{in_edge},{out_edge}')

        with open('data/user_meta_info.json', 'w') as f:
            json_to_save = {'tweets': tweets_per_userid, 'links': links_per_userid, 'threads': threads_per_userid}
            json.dump(json_to_save, f, indent=1)

        with open('data/userid_tweet_stats.csv', 'w') as f:
            f.write('userid,tweet_count,link_count,src_count,dst_count,thread_count')
            for userid, tweet_count in user_tweet_dist:
                link_count = links_per_userid.get(userid, 0)
                if type(link_count) is dict:
                    try:
                        src_count = link_count['src']['total']
                    except:
                        src_count = 0
                    try:
                        dst_count = link_count['dst']['total']
                    except:
                        dst_count = 0
                    link_count = src_count + dst_count
                thread_count = len(threads_per_userid.get(userid, []))
                f.write(f'\n{userid},{tweet_count},{link_count},{src_count},{dst_count},{thread_count}')