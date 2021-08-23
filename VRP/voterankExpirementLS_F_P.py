
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
    # index_sum = 0
    for i in result:
        result_topk.append(i[0])
        # index_sum += 1
    # if index_sum == topk:
    #     print('topk equals to len(result)')
    # elif index_sum > topk:
    #     print('topk less than len(result)')
    # else:
    #     print('error!')
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
def compute_probability(G):
    """compute the infection probability
    # Arguments
        Source_G: a graph as networkx Graph
    Returns
        the infection probability computed by  formula: <k> / (<k^2> - <k>)
    """


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

    # rank1 = []
    # for node, score in degree_rank:
    #     rank1.append(node)
    #     if len(rank1) == topk:
    #         for i in range(len(rank1)):
    #             rank1[i] = (rank1[i], ' ')
    #         return rank1
    return degree_rank[: topk]
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
def mean_value(a):
    n = len(a)
    sum_mean = 0
    for i in a:
        sum_mean += i
    return sum_mean / n


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
# edgelist = ['PGP', 'condmat']
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

    topK = [int(0.005 * len(G)), int(0.010 * len(G)), int(0.015 * len(G)), int(0.02 * len(G)),  int(0.025 * len(G)), int(0.03 * len(G))]
    nOfh_index = 1
    order = 2
    lambdaa = 0.1
    avg = 100
    muc = compute_probability(G)
    infect_prob = 1.5 * muc
    # cover_prob = mu_c
    cover_prob = muc
    # cover_prob = 0.8
    max_iter = 1000


    voterank_li = voterank(G)
    n = len(voterank_li)

    voterank_re = [(voterank_li[i], voterank_li[i]) for i in range(n)]

    h = H_index.calcHIndexValues(G, 0)  # something related to k_shell method
    I_dicOri = k_shell_entroph.originalinforEntro(G, nodes, degree_dic)
    k_shellsets = k_shell_entroph.k_shell(g1)
    resultOri = k_shell_entroph.newmethod(k_shellsets, I_dicOri)
    impro_k_shell_re = [(resultOri[i], resultOri[i]) for i in range(n)]

    dr = degree(G, int(0.03 * len(G)))  # 先找到最大比例的结果R = [(node,value),...]
    kr = kshell(G, int(0.03 * len(G)))
    iksr = impro_k_shell_re[: int(0.03 * len(G))]
    hr = h_indexs(G, nOfh_index, int(0.03 * len(G)))
    vr = voterank_re[: int(0.03 * len(G))]
    mr = MCDE.MCDE(G)[:int(0.03 * len(G))]
    ecr = ECRM.ECRM(G, hierarchy)[: int(0.03 * len(G))]
    er = EnRenewRank(G, int(0.03 * len(G)), order)
    vpr = voterank_plus.voterank_plus(G, int(0.03 * len(G)), lambdaa)

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
    n = len(G)
    for i in range(len(topK)):
        p = topK[i]

        if p == 0:
            degree_mere_li.append([0])
            k_shell_mere_li.append([0])
            iks_mere_li.append([0])
            hindex_mere_li.append([0])
            voterak_mere_li.append([0])
            MCDE_mere_li.append([0])
            ECRM_mere_li.append([0])
            EnRenew_mere_li.append([0])
            voterank_plus_mere_li.append([0])
            # new_mere_li.append([0])
        else:

            degree_mere_li.append(get_sir_result(G, dr[:p], p, avg, infect_prob, cover_prob, max_iter))
            k_shell_mere_li.append(get_sir_result(G, kr[:p], p, avg, infect_prob, cover_prob, max_iter))
            iks_mere_li.append(get_sir_result(G, iksr[:p], p, avg, infect_prob, cover_prob, max_iter))
            hindex_mere_li.append(get_sir_result(G, hr[:p], p, avg, infect_prob, cover_prob, max_iter))
            voterak_mere_li.append(get_sir_result(G, vr[:p], p, avg, infect_prob, cover_prob, max_iter))
            MCDE_mere_li.append(get_sir_result(G, mr[:p], p, avg, infect_prob, cover_prob, max_iter))
            ECRM_mere_li.append(get_sir_result(G, ecr[:p], p, avg, infect_prob, cover_prob, max_iter))
            EnRenew_mere_li.append(get_sir_result(G, er[:p], p, avg, infect_prob, cover_prob, max_iter))
            voterank_plus_mere_li.append(get_sir_result(G, vpr[:p], p, avg, infect_prob, cover_prob, max_iter))
            # new_mere_li.append(get_sir_result(G, secondMethod.newMethod(G, p), p, avg, infect_prob, cover_prob, max_iter))





    print(graph, '迭代了', avg, '次后得到的每种方法每个时刻平均感染的节点数目：(图中节点数目)', len(G))

    from pandas import DataFrame


    degree_ls = []
    k_shell_ls = []
    iks_ls = []
    h_index_ls = []
    voterank_ls = []
    MCDE_ls = []
    ECRM_ls = []
    EnRenew_ls = []
    Voterank_plus_ls = []
    # new_ls = []
    for i in range(len(topK)):
        p = topK[i]
        if p == 0:
            degree_ls.append(0)
            k_shell_ls.append(0)
            iks_ls.append(0)
            h_index_ls.append(0)
            voterank_ls.append(0)
            MCDE_ls.append(0)
            ECRM_ls.append(0)
            EnRenew_ls.append(0)
            Voterank_plus_ls.append(0)
            # new_ls.append(0)
        else:
            degree_ls.append(get_ls(G, get_topk(dr[:p], p)))
            k_shell_ls.append(get_ls(G, get_topk(kr[:p], p)))
            iks_ls.append(get_ls(G, get_topk(iksr[:p], p)))
            h_index_ls.append(get_ls(G, get_topk(hr[:p], p)))
            voterank_ls.append(get_ls(G, get_topk(vr[:p], p)))
            MCDE_ls.append(get_ls(G, get_topk(mr[:p], p)))
            ECRM_ls.append(get_ls(G, get_topk(ecr[:p], p)))
            EnRenew_ls.append(get_ls(G, get_topk(er[:p], p)))
            Voterank_plus_ls.append(get_ls(G, get_topk(vpr[:p], p)))
            # new_ls.append(get_ls(G, get_topk(secondMethod.newMethod(G, p), p)))

    data = {
    'degree': [i[-1] for i in degree_mere_li],
    'k_shell': [i[-1] for i in k_shell_mere_li],
    'isk': [i[-1] for i in iks_mere_li],
    'h_index': [i[-1] for i in hindex_mere_li],
    'voterank': [i[-1] for i in voterak_mere_li],
    'MCDE': [i[-1] for i in MCDE_mere_li],
    'ECRM': [i[-1] for i in ECRM_mere_li],
    'EnRenew': [i[-1] for i in EnRenew_mere_li],
    'Voterank+': [i[-1] for i in voterank_plus_mere_li],

    'degree_ls': degree_ls,
    'k_shell_ls': k_shell_ls,
    'isk_ls': iks_ls,
    'h_index_ls': h_index_ls,
    'voterank_ls': voterank_ls,
    'MCDE_ls': MCDE_ls,
    'ECRM_ls': ECRM_ls,
    'EnRenew_ls': EnRenew_ls,
    'Voterank+_ls': Voterank_plus_ls
    }


    df = DataFrame(data)

    df.to_excel(graph+'-F_p_ls'+'.xlsx')




