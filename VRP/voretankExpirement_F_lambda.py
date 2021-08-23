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
# import math
# import H_index
# import k_shell_entroph
# import ECRM
# import MCDE

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

def get_keys(d, va):
    for k, v in d.items():
        if va in v:
            return k

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

def mean_value(a):
    n = len(a)
    sum_mean = 0
    for i in a:
        sum_mean += i
    return sum_mean / n

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

# for i in edgelist:
#     G = nx.read_edgelist(edgelist + '.edgelist', nodetype=int)

edgelist = ['Facebook', 'Gowalla', 'dblp', 'amazon']

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


    topK = int(0.03 * len(G))
    nOfh_index = 1
    order = 2
    lambdaa = [0.1, 0.2, 0.3, 0.4, 0.5]  # F & lambda
    avg = 100
    muc = compute_probability(G)
    infect_prob = 1.5 * muc
    # cover_prob = mu_c
    cover_prob = muc
    # cover_prob = 0.8
    max_iter = 1000




    voterank_plus_result_li = []  # a result container
    for i in range(len(lambdaa)):
        voterank_plus_result_li.append(voterank_plus.voterank_plus(G, topK, lambdaa[i]))

    voterank_plus_mere_li = []
    for i in range(len(lambdaa)):

        voterank_plus_mere_li.append(get_sir_result(G, voterank_plus_result_li[i], topK, avg, infect_prob, cover_prob, max_iter))




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

    mix_li = [voterank_plus_mere_li[0], voterank_plus_mere_li[1], voterank_plus_mere_li[2], voterank_plus_mere_li[3], voterank_plus_mere_li[4]]
    new_mix = compeleteList(mix_li)
    # mix_li = [degree_meanresult, k_shell_meanresult, impr_k_shell_meanresult, h_index_meanresult, voterank_meanresult, MCDE_meanresult, ECRM_meanresult, EnRenew_meanresult, voterank_plus_meansult]
    # new_mix = compeleteList(mix_li)

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

    '0.1': new_mix[0],
    '0.2': new_mix[1],
    '0.3': new_mix[2],
    '0.4': new_mix[3],
    '0.5': new_mix[4]

    }


    df = DataFrame(data)

    df.to_excel(graph+'F_lambda'+'.xlsx')




