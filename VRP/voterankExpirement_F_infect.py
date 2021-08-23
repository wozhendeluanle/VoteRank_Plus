# Copyright (c) 2019 Chungu Guo. All rights reserved.
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
from tqdm import tqdm
import copy
import random
import voterank_plus
import math
import H_index
import k_shell_entroph
import ECRM
import MCDE
# import secondMethod

def findmaxi(a):
    maxv = max(a)
    for i in range(len(a)):
        if a[i] == maxv:
            return i+1
def get_topk(result, topk):
    """return the topk nodes
    # Arguments
        result: a list of result, [(node1, centrality), (node1, centrality), ...]
        topk: how much node will be returned
    Returns
        topk nodes as a list, [node1, node2, ...]
    """
    result_topk = []
    for i in result:
        result_topk.append(i[0])
    return result_topk
def get_sir_result(G, rank, topk, avg, infect_prob, cover_prob, max_iter):
    """perform SIR simulation
    # Arguments
        G: a graph as networkx Graph
        rank: the initial node set to simulate, [(node1, centrality), (node1, centrality), ...]我改成了[node1, node2,...]
        topk: use the topk nodes in rank to simulate
        avg: simulation times, multiple simulation to averaging
        infect_prob: the infection probability
        cover_prob: the cover probability,
        max_iter: maximum number of simulation steps
    Returns
        average simulation result, a 1-D array, indicates the scale of the spread of each step
    """
    time_num_dict_list = []  # 每一时刻感染的节点数目
    time_list = []  # 时刻

    for i in range(avg):
        time, time_num_dict = SIR(G, get_topk(rank, topk), infect_prob, cover_prob, max_iter)
        time_num_dict_list.append((list(time_num_dict.values())))
        time_list.append(time)

    max_time = max(time_list) + 1
    result_matrix = np.zeros((len(time_num_dict_list), max_time))
    for index, (row, time_num_dict) in enumerate(zip(result_matrix, time_num_dict_list)):
        row[:] = time_num_dict[-1]
        row[0:len(time_num_dict)] = time_num_dict
        result_matrix[index] = row
    result = np.mean(result_matrix, axis=0)
    average_simulation_result = []
    for i in result:
        average_simulation_result.append(i / len(G))
    return average_simulation_result
def compute_probability(Source_G):
    """compute the infection probability
    # Arguments
        Source_G: a graph as networkx Graph
    Returns
        the infection probability computed by  formula: <k> / (<k^2> - <k>)
    """
    G = nx.Graph()
    G = Source_G
    degree_dict = dict(G.degree())
    k = 0.0
    k_pow = 0.0
    for i in degree_dict:
        k = k + degree_dict[i]
        k_pow = k_pow + degree_dict[i] * degree_dict[i]

    k = k / G.number_of_nodes()
    k_pow = k_pow / G.number_of_nodes()
    pro = k / (k_pow - k)
    return pro
def SIR(g, infeacted_set, infect_prob, cover_prob, max_iter):
    """Perform once simulation
    # Arguments
        g: a graph as networkx Graph
        infeacted_set: the initial node set to simulate, [node1, node2, ...]
        infect_prob: the infection probability
        cover_prob : the cover probability,
        max_iter: maximum number of simulation steps
    Returns
        time: the max time step in this simulation
        time_count_dict: record the scale of infection at each step, {1:5, 2:20, ..., time: scale, ...}
    """
    time = 0
    time_count_dict = {}
    time_count_dict[time] = len(infeacted_set)
    # infeacted_set = infeacted_set
    node_state = {}
    covered_set = set()

    for node in nx.nodes(g):
        if node in infeacted_set:
            node_state[node] = 'i'
        else:
            node_state[node] = 's'

    while len(infeacted_set) != 0 and max_iter != 0:
        ready_to_cover = []
        ready_to_infeact = []
        for node in infeacted_set:
            nbrs = list(nx.neighbors(g, node))
            nbr = np.random.choice(nbrs)
            if random.uniform(0, 1) <= infect_prob and node_state[nbr] == 's':
                node_state[nbr] = 'i'
                ready_to_infeact.append(nbr)
            if random.uniform(0, 1) <= cover_prob:
                ready_to_cover.append(node)
        for node in ready_to_cover:
            node_state[node] = 'r'
            infeacted_set.remove(node)
            covered_set.add(node)
        for node in ready_to_infeact:
            infeacted_set.append(node)
        max_iter -= 1
        print('这是第' + str(max_iter) + '次迭代')
        time += 1
        time_count_dict[time] = len(covered_set) + len(infeacted_set)
    return time, time_count_dict
def get_ls(g, infeacted_set):
    """compute the average shortest path in the initial node set
     # Arguments
         g: a graph as networkx Graph
         infeacted_set: the initial node set(list)
     Returns
         return the average shortest path
     """
    dis_sum = 0
    path_num = 0
    S = len(infeacted_set)
    for u in infeacted_set:
        for v in infeacted_set:
            if u != v:
                try:
                    dis_sum += nx.shortest_path_length(g, u, v)
                    path_num += 1
                except:
                    dis_sum += 0
                    path_num -= 1
    return dis_sum / S * (S - 1)
def compeleteList(mix_li):
    '''

    :param mix_li: a = [],b =[], mix_li = [a, b]
    :return: mix_li = [a,b] 其中每个列表长度相同，补最大值操作
    '''

    length_li = []
    for i in mix_li:
        length_li.append(len(i))
    max_len = max(length_li)

    for i in mix_li:
        if len(i) < max_len:
            i.extend([max(i)] * (max_len - len(i)))
    return mix_li
def h_indexs(G, nOfh_index, topk):
    nodes = list(nx.nodes(G))
    n = len(nodes)
    h = H_index.calcHIndexValues(G, nOfh_index)
    h_list = [(nodes[i], h[i]) for i in range(n)]  # (alt, imp): (候选元素，重要性)
    h_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return h_list[:topk]
def degree(g, topk):
    """use the degree to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by degree, [(node1, ' '), (node2, '' ), ...]
     """
    degree_rank = nx.degree_centrality(g)
    degree_rank = sorted(degree_rank.items(), key=lambda x: x[1], reverse=True)
    rank1 = []
    for node, score in degree_rank:
        rank1.append(node)
        if len(rank1) == topk:
            for i in range(len(rank1)):
                rank1[i] = (rank1[i], ' ')
            return rank1
    return rank1
def voterank(G, number_of_nodes=None, max_iter=200):
    """Compute a list of seeds for the nodes in the graph using VoteRank

    VoteRank [1]_ computes a ranking of the nodes in the graph G based on a voting
    scheme. With VoteRank, all nodes vote for each neighbours and the node with
    the highest score is elected iteratively. The voting ability of neighbors of
    elected nodes will be decreased in subsequent turn.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    number_of_nodes : integer, optional
        Number of ranked nodes to extract (default all nodes).

    max_iter : integer, optional
        Maximum number of iterations to rank nodes.

    Returns
    -------
    voterank : list
        Ordered list of computed seeds.

    Raises
    ------
    NetworkXNotImplemented:
        If G is digraph.

    References
    ----------
    .. [1] Zhang, J.-X. et al. (2016).
        Identifying a set of influential spreaders in complex networks.
        Sci. Rep. 6, 27823; doi: 10.1038/srep27823.
    """
    voterank = []
    if len(G) == 0:
        return voterank
    if number_of_nodes is None or number_of_nodes > len(G):
        number_of_nodes = len(G)
    avgDegree = sum(deg for _, deg in G.degree()) / float(len(G))
    # step 1 - initiate all nodes to (0,1) (score, voting ability)
    for _, v in G.nodes(data=True):
        v['voterank'] = [0, 1]
    # Repeat steps 1b to 4 until num_seeds are elected.
    for _ in range(max_iter):
        # step 1b - reset rank
        for _, v in G.nodes(data=True):
            v['voterank'][0] = 0
        # step 2 - vote
        for n, nbr in G.edges():
            G.nodes[n]['voterank'][0] += G.nodes[nbr]['voterank'][1]
            G.nodes[nbr]['voterank'][0] += G.nodes[n]['voterank'][1]
        for n in voterank:
            G.nodes[n]['voterank'][0] = 0
        # step 3 - select top node
        n, value = max(G.nodes(data=True), key=lambda x: x[1]['voterank'][0])
        if value['voterank'][0] == 0:
            return voterank
        voterank.append(n)
        if len(voterank) >= number_of_nodes:
            return voterank
        # weaken the selected node
        G.nodes[n]['voterank'] = [0, 0]
        # step 4 - update voterank properties
        for nbr in G.neighbors(n):
            G.nodes[nbr]['voterank'][1] -= 1 / avgDegree
    return voterank
def kshell(G, topk):
    """use the kshell to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by kshell, [(node1, ' '), (node2, ' '), ...]
     """
    node_core = nx.core_number(G)
    core_node_list = {}
    for node in node_core:
        if node_core[node] not in core_node_list:
            core_node_list[node_core[node]] = []
        core_node_list[node_core[node]].append((node, nx.degree(G, node)))

    for core in core_node_list:
        core_node_list[core] = sorted(core_node_list[core], key=lambda x: x[1], reverse=True)
    core_node_list = sorted(core_node_list.items(), key=lambda x: x[0], reverse=True)
    kshellrank = []
    for core, node_list in core_node_list:
        kshellrank.extend([n[0] for n in node_list])

    rank = []
    for node in kshellrank:
        rank.append((node, ' '))
        if len(rank) == topk:
            return rank
def EnRenewRank(G, topk, order):
    """use the our method to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by EnRenewRank, [(node1, score), (node2, score), ...]
     """

    all_degree = nx.number_of_nodes(G) - 1
    k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
    k_entropy = - k_ * ((k_ / all_degree) * math.log((k_ / all_degree)))

    # node's information pi
    node_information = {}
    for node in nx.nodes(G):
        information = (G.degree(node) / all_degree)
        node_information[node] = - information * math.log(information)

    # node's entropy Ei
    node_entropy = {}
    for node in nx.nodes(G):
        node_entropy[node] = 0
        for nbr in nx.neighbors(G, node):
            node_entropy[node] += node_information[nbr]

    rank = []
    for i in range(topk):
        # choose the max entropy node
        max_entropy_node, entropy = max(node_entropy.items(), key=lambda x: x[1])
        rank.append((max_entropy_node, entropy))

        cur_nbrs = nx.neighbors(G, max_entropy_node)
        for o in range(order):
            for nbr in cur_nbrs:
                if nbr in node_entropy:
                    node_entropy[nbr] -= (node_information[max_entropy_node] / k_entropy) / (2 ** o)
            next_nbrs = []
            for node in cur_nbrs:
                nbrs = nx.neighbors(G, node)
                next_nbrs.extend(nbrs)
            cur_nbrs = next_nbrs

        # set the information quantity of selected nodes to 0
        node_information[max_entropy_node] = 0
        # delete max_entropy_node
        node_entropy.pop(max_entropy_node)
    return rank
def get_keys(d, va):
    for k, v in d.items():
        if va in v:
            return k
def k_shell(graph):
    importance_dict = {}
    level=1
    while len(graph.degree):
        importance_dict[level]=[]
        while True:
            level_node_list=[]
            for item in graph.degree:
                if item[1]<=level:
                    level_node_list.append(item[0])
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree):
                return importance_dict
            if min(graph.degree,key=lambda x:x[1])[1]>level:
                break
        level=min(graph.degree,key=lambda x:x[1])[1]
    return importance_dict
def get_weight(G, degree_dic):  # wij表示：i给j投票的能力，wij ！= wji
    weight = {}
    nodes = nx.nodes(G)
    for node in nodes:
        sum1 = 0
        for nbr in nx.neighbors(G, node):
            sum1 += degree_dic[nbr]
        for neigh in nx.neighbors(G, node):
            weight[(node, neigh)] = degree_dic[neigh] / sum1
    # for i, j in nx.edges(G):
    #     sum1 = 0
    #     for nbr in nx.neighbors(G, i):
    #         sum1 += degree_dic[nbr]
    #     weight[(i, j)] = degree_dic[j] / sum1
    return weight
def get_node_score(G, nodesNeedcalcu, node_ability, degree_dic):

    weight = get_weight(G, degree_dic)
    node_score = {}
    for node in nodesNeedcalcu:  # for ever node add the neighbor's weighted ability
        sum2 = 0
        for nbr in nx.neighbors(G, node):
            sum2 += node_ability[nbr] * weight[(nbr, node)]
        node_score[node] = sum2
    return node_score
def get_node_score2(G, nodesNeedcalcu, node_ability, degree_dic):

    weight = get_weight(G, degree_dic)
    node_score = {}
    for node in nodesNeedcalcu:  # for ever node add the neighbor's weighted ability
        sum2 = 0
        neighbors = list(nx.neighbors(G, node))
        for nbr in neighbors:
            sum2 += node_ability[nbr] * weight[(nbr, node)]
        node_score[node] = math.sqrt(len(neighbors) * sum2)
    return node_score
# def ournewRank(G, l, lambdaa):
#     '''
#
#     :param G: use new indicator + lambda + voterank, the vote ability = dij
#     :param l: the number of spreaders
#     :param lambdaa: retard infactor
#     :return:
#     '''
#     # count dict
#     nodes = list(nx.nodes(G))
#     degree_li = nx.degree(G)
#     degree_dic = {}
#     for i in degree_li:
#         degree_dic[i[0]] = i[1]
#
#     # node's vote information
#     node_ability = {}
#     for i in degree_li:
#         node_ability[i[0]] = i[1]
#
#     # node's score
#     node_score = get_node_score(G, nodes, node_ability, degree_dic)
#
#     rank = []
#     for i in range(l):
#         # choose the max entropy node
#         max_score_node, score = max(node_score.items(), key=lambda x: x[1])
#         rank.append((max_score_node, score))
#         # set the information quantity of selected nodes to 0
#         node_ability[max_score_node] = 0
#         # set entropy to 0
#         node_score.pop(max_score_node)
#         # node_score[max_score_node] = 0
#         # nodes.remove(max_score_node)
#
#
#         cur_nbrs = list(nx.neighbors(G, max_score_node))  #for the max score node's neighbor
#         for nbr in cur_nbrs:  # retard the vote ability
#             node_ability[nbr] = node_ability[nbr] * lambdaa
#
#         H = []
#         H.extend(cur_nbrs)
#         for i in cur_nbrs:  # find the neighbor and neighbor's neighbor
#             nbrs = nx.neighbors(G, i)
#             H.extend(nbrs)
#
#         H = list(set(H))
#         for i in rank:
#             if i[0] in H:
#                 H.remove(i[0])
#         new_nodeScore = get_node_score(G, H, node_ability, degree_dic)
#         for k in new_nodeScore.keys():
#             node_score[k] = new_nodeScore[k]
#         # node_score.pop(max_score_node)
#
#         # #set the information quantity of selected nodes to 0
#         # node_ability[max_score_node] = 0
#         # # set entropy to 0
#         # node_score.pop(max_score_node)
#         # print(i, rank)
#     return rank
def mean_value(a):
    n = len(a)
    sum_mean = 0
    for i in a:
        sum_mean += i
    return sum_mean / n
# def ournewRank2(G, l, order):
#     '''
#
#     :param G: use new indicator + l-reachable + voterank, the vote ability = dij
#     :param l: the number of spreaders
#     :param order: l-reachable neighbors
#     :return:
#     '''
#     # count dict
#     # N - 1
#     all_degree = nx.number_of_nodes(G) - 1
#     # avg degree
#     k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
#     # E<k>
#     k_entropy = - k_ * ((k_ / all_degree) * math.log((k_ / all_degree)))
#     nodes = list(nx.nodes(G))
#     degree_li = nx.degree(G)
#     degree_dic = {}
#     for i in degree_li:
#         degree_dic[i[0]] = i[1]
#
#     # node's vote information
#     node_ability = {}
#     for i in degree_li:
#         node_ability[i[0]] = i[1]
#
#     # node's score
#     node_score = get_node_score(G, nodes, node_ability, degree_dic)
#
#     rank = []
#     for i in range(l):
#         # choose the max entropy node
#         max_score_node, score = max(node_score.items(), key=lambda x: x[1])
#         rank.append((max_score_node, score))
#         # set the information quantity of selected nodes to 0
#         node_ability[max_score_node] = 0
#         # set entropy to 0
#         node_score.pop(max_score_node)
#         # node_score[max_score_node] = 0
#         # nodes.remove(max_score_node)
#
#
#         cur_nbrs = list(nx.neighbors(G, max_score_node))  #for the max score node's neighbor
#         for o in range(int(order)):
#             for nbr in cur_nbrs:
#                 if nbr in node_ability:
#                     node_ability[nbr] -= (node_ability[max_score_node] / k_entropy) / (2 ** o)
#             next_nbrs = []
#             for node in cur_nbrs:
#                 nbrs = nx.neighbors(G, node)
#                 next_nbrs.extend(nbrs)
#             cur_nbrs = next_nbrs
#
#         H = []
#         H.extend(cur_nbrs)
#         for i in cur_nbrs:  # find the neighbor and neighbor's neighbor
#             nbrs = nx.neighbors(G, i)
#             H.extend(nbrs)
#
#         H = list(set(H))
#         for i in rank:
#             if i[0] in H:
#                 H.remove(i[0])
#         new_nodeScore = get_node_score(G, H, node_ability, degree_dic)
#         for k in new_nodeScore.keys():
#             node_score[k] = new_nodeScore[k]
#         # node_score.pop(max_score_node)
#
#         # #set the information quantity of selected nodes to 0
#         # node_ability[max_score_node] = 0
#         # # set entropy to 0
#         # node_score.pop(max_score_node)
#         # print(i, rank)
#     return rank
# def ournewRank3(G, l, lambdaa):
#     '''
#
#     :param G: use new indicator + lambda + voterank, the vote ability = log(dij)
#     :param l: the number of spreaders
#     :param lambdaa: retard infactor
#     :return:
#     '''
#     # count dict
#     nodes = list(nx.nodes(G))
#     degree_li = nx.degree(G)
#     degree_dic = {}
#     for i in degree_li:
#         degree_dic[i[0]] = i[1]
#
#     # node's vote information
#     node_ability = {}
#     for i in degree_li:
#         node_ability[i[0]] = math.log(i[1])
#
#     # node's score
#     node_score = get_node_score(G, nodes, node_ability, degree_dic)
#
#     rank = []
#     for i in range(l):
#         # choose the max entropy node
#         max_score_node, score = max(node_score.items(), key=lambda x: x[1])
#         rank.append((max_score_node, score))
#         # set the information quantity of selected nodes to 0
#         node_ability[max_score_node] = 0
#         # set entropy to 0
#         node_score.pop(max_score_node)
#         # node_score[max_score_node] = 0
#         # nodes.remove(max_score_node)
#
#
#         cur_nbrs = list(nx.neighbors(G, max_score_node))  #for the max score node's neighbor
#         for nbr in cur_nbrs:  # retard the vote ability
#             node_ability[nbr] = node_ability[nbr] * lambdaa
#
#         H = []
#         H.extend(cur_nbrs)
#         for i in cur_nbrs:  # find the neighbor and neighbor's neighbor
#             nbrs = nx.neighbors(G, i)
#             H.extend(nbrs)
#
#         H = list(set(H))
#         for i in rank:
#             if i[0] in H:
#                 H.remove(i[0])
#         new_nodeScore = get_node_score(G, H, node_ability, degree_dic)
#         for k in new_nodeScore.keys():
#             node_score[k] = new_nodeScore[k]
#         # node_score.pop(max_score_node)
#
#         # #set the information quantity of selected nodes to 0
#         # node_ability[max_score_node] = 0
#         # # set entropy to 0
#         # node_score.pop(max_score_node)
#         # print(i, rank)
#     return rank


# G = nx.Graph()
# G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# G.add_edges_from([(0, 1), (0, 10), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (3, 4), (3, 9), (4, 7), (5, 6), (5, 8), (8, 9),(7, 10)])
# G = nx.Graph()
# G.add_nodes_from(list(range(1, 27)))
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 11), (2, 12), (3, 4), (3, 8), (4, 23), (5, 9), (5, 10), (6, 7), (8, 26), (8, 25), (13, 15), (14, 15), (15, 17), (17, 23), (16, 23), (18, 23), (19, 23), (21, 23), (22, 23), (20, 23), (24, 25)])

# edgelist = 'dolphins'
# edgelist = 'CEnew'
# edgelist = 'jazz'
# edgelist = 'router'
# edgelist = 'yeast'
# edgelist = 'power'
# edgelist = 'email'
# edgelist = 'USAir97'
# edgelist = 'USAir2010'
# edgelist = 'hamster'


edgelist = ['Facebook']
# edgelist = ['PGP', 'condmat', 'Facebook', 'Gowalla', 'dblp', 'amazon']
for graph in edgelist:
    print('Now it\'s ' + graph)
    G = nx.read_edgelist(graph + '.edgelist', nodetype=int)
# G = nx.read_edgelist(edgelist + '.edgelist', nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    g = G.copy()
    g1 = G.copy()


    nodes = list(nx.nodes(G))
    degree_dic = {}
    for i in nodes:
        degree_dic[i] = nx.degree(G, i)
    degreelist = [i[1] for i in nx.degree(G)]

    mean_k = mean_value(degreelist)
    mu_c = mean_k / (mean_k ** 2 - mean_k)
    hierarchy = ECRM.ok_shell(g)  # ECRM method used

    topK = int(0.03 * len(G))
    nOfh_index = 1
    order = 2
    lambdaa = 0.1
    avg = 100
    infect_prob = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]  # F & infect probability
    cover_prob = compute_probability(G)
    # cover_prob = 0.8

    max_iter = 2000


    voterank_li = voterank(G)
    n = len(voterank_li)

    voterank_re = [(voterank_li[i], voterank_li[i]) for i in range(n)]

    h = H_index.calcHIndexValues(G, 0)  # something related to k_shell method
    I_dicOri = k_shell_entroph.originalinforEntro(G, nodes, degree_dic)
    k_shellsets = k_shell_entroph.k_shell(g1)
    resultOri = k_shell_entroph.newmethod(k_shellsets, I_dicOri)
    impro_k_shell_re = [(resultOri[i], resultOri[i]) for i in range(n)]

    degree_result = degree(G, topK)
    voterank_result = voterank_re[: topK]
    impr_k_shell_result = impro_k_shell_re[: topK]
    h_index_result = h_indexs(G, nOfh_index, topK)
    kshell_result = kshell(G, topK)
    MCDE_result = MCDE.MCDE(G)[:topK]
    ECRM_result = ECRM.ECRM(G, hierarchy)[: topK]
    EnRenewRank_result = EnRenewRank(G, topK, order)
    voterank_plus_result = voterank_plus.voterank_plus(G, topK, lambdaa)
    # new_result = secondMethod.newMethod(G, topK)
    # ournewRank_result = ournewRank(G, topK, lambdaa)
    # ournewRank2_result = ournewRank2(G, topK, order)
    # ournewRank3_result = ournewRank3(G, topK, lambdaa)


    # ournewRank_result = ournewRank(G, topK, lambdaa)
    # ournewRank2_result = ournewRank2(G, topK, order)
    # ournewRank3_result = ournewRank3(G, topK, lambdaa)

    degree_mere_li = []
    voterak_mere_li = []
    iks_mere_li = []
    hindex_mere_li = []
    k_shell_mere_li = []
    MCDE_mere_li = []
    ECRM_mere_li = []
    EnRenew_mere_li = []
    voterank_plus_mere_li = []
    # new_mere_li = []
    for i in range(len(infect_prob)):
        degree_mere_li.append(get_sir_result(G, degree_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        k_shell_mere_li.append(get_sir_result(G, kshell_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        iks_mere_li.append(get_sir_result(G, impr_k_shell_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        hindex_mere_li.append(get_sir_result(G, h_index_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        voterak_mere_li.append(get_sir_result(G, voterank_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        MCDE_mere_li.append(get_sir_result(G, MCDE_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        ECRM_mere_li.append(get_sir_result(G, ECRM_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        EnRenew_mere_li.append(get_sir_result(G, EnRenewRank_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        voterank_plus_mere_li.append(get_sir_result(G, voterank_plus_result, topK, avg, infect_prob[i], cover_prob, max_iter))
        # new_mere_li.append(get_sir_result(G, new_result, topK, avg, infect_prob[i], cover_prob, max_iter))




    # degree_meanresult = get_sir_result(G, degree_result, topK, avg, infect_prob, cover_prob, max_iter)
    # k_shell_meanresult = get_sir_result(G, kshell_result, topK, avg, infect_prob, cover_prob, max_iter)
    # impr_k_shell_meanresult = get_sir_result(G, impr_k_shell_result, topK, avg, infect_prob, cover_prob, max_iter)
    # h_index_meanresult = get_sir_result(G, h_index_result, topK, avg, infect_prob, cover_prob, max_iter)
    # voterank_meanresult = get_sir_result(G, voterank_result, topK, avg, infect_prob, cover_prob, max_iter)
    # MCDE_meanresult  = get_sir_result(G, MCDE_result, topK, avg,  infect_prob, cover_prob, max_iter)
    # ECRM_meanresult = get_sir_result(G, ECRM_result, topK, avg,  infect_prob, cover_prob, max_iter)
    # EnRenew_meanresult = get_sir_result(G, EnRenewRank_result, topK, avg,  infect_prob, cover_prob, max_iter)
    # # ournewrank_meansult = get_sir_result(G, ournewRank_result, topK, avg, infect_prob, cover_prob, max_iter)
    # # ournewrank2_meansult = get_sir_result(G, ournewRank2_result, topK, avg, infect_prob, cover_prob, max_iter)
    # # ournewrank3_meansult = get_sir_result(G, ournewRank3_result, topK, avg, infect_prob, cover_prob, max_iter)
    # voterank_plus_meansult = get_sir_result(G, voterank_plus_result, topK, avg, infect_prob, cover_prob, max_iter)

    # mix_li = [degree_meanresult, k_shell_meanresult, impr_k_shell_meanresult, h_index_meanresult, voterank_meanresult, MCDE_meanresult, ECRM_meanresult, EnRenew_meanresult, voterank_plus_meansult]
    # new_mix = compeleteList(mix_li)
    print(graph, '迭代了', avg, '次后得到的每种方法每个时刻平均感染的节点数目：(图中节点数目)', len(G))
    # print('degree_non', '用时:', len(degree_non_meanresult), '   达到最大感染数目用时:', findmaxi(degree_non_meanresult), '   最大感染数目:', max(degree_non_meanresult), degree_non_meanresult)
    # print('degree', '用时', len(degree_meanresult),  '   达到最大感染数目用时:', findmaxi(degree_meanresult), '   最大感染数目:', max(degree_meanresult), degree_meanresult)
    # print('kshell_non', '用时', len(k_shell_non_meanresult),  '   达到最大感染数目用时:', findmaxi(k_shell_non_meanresult), '   最大感染数目:', max(k_shell_non_meanresult), k_shell_non_meanresult)
    # print('kshell', '用时', len(k_shell_meanresult),  '   达到最大感染数目用时:', findmaxi(k_shell_meanresult), '   最大感染数目:', max(k_shell_meanresult), k_shell_meanresult)
    # print('voterank_non', '用时', len(voterank_non_meanresult),  '   达到最大感染数目用时:', findmaxi(voterank_non_meanresult), '   最大感染数目:', max(voterank_non_meanresult), voterank_non_meanresult)
    # print('voterank', '用时', len(voterank_meanresult),  '   达到最大感染数目用时:', findmaxi(voterank_meanresult), '   最大感染数目:', max(voterank_meanresult), voterank_meanresult)
    # print('EnRenew', '用时', len(EnRenew_meanresult),  '   达到最大感染数目用时:', findmaxi(EnRenew_meanresult), '   最大感染数目:', max(EnRenew_meanresult), EnRenew_meanresult)
    # print('ournewRank', '用时', len(ournewrank_meansult),  '   达到最大感染数目用时:', findmaxi(ournewrank_meansult), '   最大感染数目:', max(ournewrank_meansult), ournewrank_meansult)
    # print('ournewRank2', '用时', len(ournewrank2_meansult),  '   达到最大感染数目用时:', findmaxi(ournewrank2_meansult), '   最大感染数目:', max(ournewrank2_meansult), ournewrank2_meansult)
    # print('ournewRank3', '用时', len(ournewrank3_meansult),  '   达到最大感染数目用时:', findmaxi(ournewrank3_meansult), '   最大感染数目:', max(ournewrank3_meansult), ournewrank3_meansult)

    from pandas import DataFrame

    # data={
    #
    # 'method':['degree', 'impr_k_shell', 'h_index', 'kshell', 'voterank', 'MCDE', 'ECRM', 'EnRenew', 'ournewrank', 'ournewrank2', 'ournewrank3', 'ourrank4'],
    #
    # '用时':[len(degree_meanresult), len(impr_k_shell_meanresult), len(h_index_meanresult), len(k_shell_meanresult), len(voterank_meanresult), len(MCDE_meanresult), len(ECRM_meanresult), len(EnRenew_meanresult), len(ournewrank_meansult), len(ournewrank2_meansult), len(ournewrank3_meansult), len(voterank_plus_meansult)],
    #
    # '达到最大节点用时':[findmaxi(degree_meanresult), findmaxi(impr_k_shell_meanresult), findmaxi(h_index_meanresult), findmaxi(k_shell_meanresult),  findmaxi(voterank_meanresult), findmaxi(MCDE_meanresult), findmaxi(ECRM_meanresult), findmaxi(EnRenew_meanresult), findmaxi(ournewrank_meansult), findmaxi(ournewrank2_meansult), findmaxi(ournewrank3_meansult), findmaxi(voterank_plus_meansult)],
    # '最大感染数目':[max(degree_meanresult), max(impr_k_shell_meanresult), max(h_index_meanresult), max(k_shell_meanresult), max(voterank_meanresult), max(MCDE_meanresult), max(ECRM_meanresult), max(EnRenew_meanresult), max(ournewrank_meansult), max(ournewrank2_meansult), max(ournewrank3_meansult), max(voterank_plus_meansult)],
    # '数据结果':[degree_meanresult, impr_k_shell_meanresult, h_index_meanresult, k_shell_meanresult,  voterank_meanresult, MCDE_meanresult, ECRM_meanresult, EnRenew_meanresult, ournewrank_meansult, ournewrank2_meansult, ournewrank3_meansult, voterank_plus_meansult],
    # 'Ls':[get_ls(G, get_topk(degree_result, topK)), get_ls(G, get_topk(impr_k_shell_result, topK)), get_ls(G, get_topk(h_index_result, topK)), get_ls(G, get_topk(kshell_result, topK)), get_ls(G, get_topk(voterank_result, topK)), get_ls(G, get_topk(MCDE_result, topK)),  get_ls(G, get_topk(ECRM_result, topK)), get_ls(G, get_topk(EnRenewRank_result, topK)), get_ls(G, get_topk(ournewRank_result, topK)), get_ls(G, get_topk(ournewRank2_result, topK)), get_ls(G, get_topk(ournewRank3_result, topK)), get_ls(G, get_topk(voterank_plus_result, topK))],
    #
    # }

    data = {
    'degree': [i[-1] for i in degree_mere_li],
    'k_shell': [i[-1] for i in k_shell_mere_li],
    'isk': [i[-1] for i in iks_mere_li],
    'h_index': [i[-1] for i in hindex_mere_li],
    'voterank': [i[-1] for i in voterak_mere_li],
    'MCDE': [i[-1] for i in MCDE_mere_li],
    'ECRM': [i[-1] for i in ECRM_mere_li],
    'EnRenew': [i[-1] for i in EnRenew_mere_li],
    'Voterank+': [i[-1] for i in voterank_plus_mere_li]
    }


    df = DataFrame(data)

    df.to_excel(graph+'F_infep'+'.xlsx')




