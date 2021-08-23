# -*- coding: utf-8 -*-
import random
import numpy as np
import voterank_plus
import math
import H_index
import k_shell_entroph
import ECRM
import MCDE
import sys
import networkx as nx
sys.path.append('..')
from Linear_Threshold_master import linear_threshold

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
def LT_Para_Get_Seeds(Method_result):
    return [i[0] for i in Method_result]
def LT_Activited_nodes(LT_Result):
    del LT_Result[0]  # don't include seed nodes
    sumOfActNodes = 0
    for i in LT_Result:
        sumOfActNodes += len(i)
    return sumOfActNodes


# edgelist = ['CEnew','email',  'jazz', 'router', 'power', 'netscience', 'USAir2010',  'yeast', 'Gowalla', 'dblp', 'condmat', 'PGP', 'amazon']
# edgelist = ['CEnew', 'email', 'hamster', 'jazz', 'router', 'power', 'netscience', 'USAir2010',  'yeast']
edgelist = ['Facebook']
for graph in edgelist:

    G = nx.read_edgelist(graph + '.edgelist', nodetype=int)  # get Graph
    G.remove_edges_from(nx.selfloop_edges(G))
    g = G.copy()
    g1 = G.copy()

    nodes = list(nx.nodes(G))
    degree_dic = {}
    for i in nodes:
        degree_dic[i] = nx.degree(G, i)
    degreelist = [i[1] for i in nx.degree(G)]


    hierarchy = ECRM.ok_shell(g)  # ECRM method used

    topK = [int(0.005 * len(G)), int(0.010 * len(G)), int(0.015 * len(G)), int(0.02 * len(G)), int(0.025 * len(G)),
            int(0.03 * len(G))]
    nOfh_index = 1
    order = 2
    lambdaa = 0.1


    voterank_li = voterank(G)
    n = len(voterank_li)

    voterank_re = [(voterank_li[i], voterank_li[i]) for i in range(n)]

    h = H_index.calcHIndexValues(G, 0)  # something related to k_shell method
    I_dicOri = k_shell_entroph.originalinforEntro(G, nodes, degree_dic)
    k_shellsets = k_shell_entroph.k_shell(g1)
    resultOri = k_shell_entroph.newmethod(k_shellsets, I_dicOri)
    impro_k_shell_re = [(resultOri[i], resultOri[i]) for i in range(n)]

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
            degree_mere_li.append(0)
            k_shell_mere_li.append(0)
            iks_mere_li.append(0)
            hindex_mere_li.append(0)
            voterak_mere_li.append(0)
            MCDE_mere_li.append(0)
            ECRM_mere_li.append(0)
            EnRenew_mere_li.append(0)
            voterank_plus_mere_li.append(0)
            # new_mere_li.append([0])
        else:
            dr = LT_Para_Get_Seeds(degree(G, p))  # dr = [node, ..., node]
            kr = LT_Para_Get_Seeds(kshell(G, p))
            iksr = LT_Para_Get_Seeds(impro_k_shell_re[: p])
            hr = LT_Para_Get_Seeds(h_indexs(G, nOfh_index, p))
            vr = LT_Para_Get_Seeds(voterank_re[: p])
            mr = LT_Para_Get_Seeds(MCDE.MCDE(G)[:p])
            ecr = LT_Para_Get_Seeds(ECRM.ECRM(G, hierarchy)[: p])
            er = LT_Para_Get_Seeds(EnRenewRank(G, p, order))
            vpr = LT_Para_Get_Seeds(voterank_plus.voterank_plus(G, p, lambdaa))


            degree_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, dr, steps=0)))
            k_shell_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, kr, steps=0)))
            iks_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, iksr, steps=0)))
            hindex_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, hr, steps=0)))
            voterak_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, vr, steps=0)))
            MCDE_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, mr, steps=0)))
            ECRM_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, ecr, steps=0)))
            EnRenew_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, er, steps=0)))
            voterank_plus_mere_li.append(LT_Activited_nodes(linear_threshold.linear_threshold(G, vpr, steps=0)))
            # new_mere_li.append(get_sir_result(G, secondMethod.newMethod(G, p), p, avg, infect_prob, cover_prob, max_iter))



    from pandas import DataFrame


    data = {
        'degree': degree_mere_li,
        'k_shell': k_shell_mere_li,
        'isk': iks_mere_li,
        'h_index': hindex_mere_li,
        'voterank': voterak_mere_li,
        'MCDE': MCDE_mere_li,
        'ECRM': ECRM_mere_li,
        'EnRenew': EnRenew_mere_li,
        'Voterank+': voterank_plus_mere_li
    }

    df = DataFrame(data)

    df.to_excel(graph + 'LT_nodes' + '.xlsx')




