import networkx as nx
import matplotlib.pyplot as plp
import math
import operator

def pair(i, j):
    if i < j:
        return (i, j)
    else:
        return (j, i)
def ok_shell(graph):
    IT = {}
    importance_dict = {}
    level = 1
    iter = 1
    while len(graph.degree):
        importance_dict[level] = []
        while True:
            level_node_list = []
            for item in graph.degree:  # item represents each node ,item[0]:nodes,item[1]:degree
                if item[1] <= level:
                    level_node_list.append(item[0])  #IT value
            IT[iter] = level_node_list
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            iter += 1
            if not len(graph.degree):
                return IT
            if min(graph.degree, key=lambda x: x[1])[1] > level:
                break
        level = min(graph.degree, key=lambda x: x[1])[1]
    return IT
def get_dic_values(d):
    values = []
    for i in d.values():
        values.extend(i)
    return values
def get_keys(d, va):
    for k, v in d.items():
        if va in v:
            return k
def S_V(G, hierarchy):  # SV = {nodes:[len of hierarchy neigh]}
    nodes = list(nx.nodes(G))
    sv = {}
    numOfHierarchy = len(hierarchy)

    # print(nodes)
    for node in nodes:
        hiera_nei = [0] * numOfHierarchy
        for nei in nx.neighbors(G, node):
            if nei in get_dic_values(hierarchy):
                index = get_keys(hierarchy, nei)
                hiera_nei[index - 1] += 1
        sv[node] = hiera_nei
    return sv
def S_V_ba(G, hierarchy):
    f = len(hierarchy)
    sv_ba = {}
    for item in nx.degree(G):
        sv_ba[item[0]] = item[1] / f
    return sv_ba
def maxdegree(G):
    degree_dic = {}
    for item in nx.degree(G):
        degree_dic[item[0]] = item[1]
    max_node, degree = max(degree_dic.items(), key=lambda x: x[1])
    return degree
def ECRM(G, hierarchy):
    nodes = list(nx.nodes(G))
    CC = {}
    svs = S_V(G, hierarchy)  # notice:k_shell will delete all the nodes
    f = len(hierarchy)
    sv_ba = S_V_ba(G, hierarchy)

    for (i, j) in nx.edges(G):
        sum_divi_son = 0
        sum_divi_mam1 = 0  # without sqrt
        sum_divi_mam2 = 0  # without sqrt
        # print(svs[i])
        for k in range(f):
            sum_divi_son += (svs[i][k]-sv_ba[i]) * (svs[j][k]-sv_ba[j])
            sum_divi_mam1 += (svs[i][k] - sv_ba[i]) ** 2
            sum_divi_mam2 += (svs[j][k] - sv_ba[j]) ** 2
        CC[pair(i, j)] = sum_divi_son / (math.sqrt(sum_divi_mam1) * math.sqrt(sum_divi_mam2))


    degree_dic = {}
    for item in nx.degree(G):
        degree_dic[item[0]] = item[1]
    max_node, d = max(degree_dic.items(), key=lambda x: x[1])

    SCC = {}
    for node in nodes:
        sum_scc = 0
        for nbr in nx.neighbors(G, node):
            sum_scc += ((2 - CC[pair(node, nbr)]) + (2 * degree_dic[nbr] / d + 1))
        SCC[node] = sum_scc

    CRM = {}
    for i in nodes:
        sum_crm = 0
        for j in nx.neighbors(G, i):
            sum_crm += SCC[j]
        CRM[i] = sum_crm

    ECRM = {}
    for i in nodes:
        sum_ecrm = 0
        for j in nx.neighbors(G, i):
            sum_ecrm += CRM[j]
        ECRM[i] = sum_ecrm
    sort_val_ECRM = sorted(ECRM.items(), key=operator.itemgetter(1), reverse=True)
    return sort_val_ECRM



# G = nx.Graph()
# G.add_nodes_from(list(range(1, 18)))
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (1, 6), (2, 3), (2, 4), (3, 4), (5, 6), (5, 7), (6, 7), (7, 8), (7, 9), (8, 9), (8, 16), (9, 10), (9, 11), (10, 12), (10, 14), (11, 15), (12, 13), (12, 16),  (17, 16)])
# g = G.copy()  # must have a copy of original graph
# hierarchy = ok_shell(g)  # the g can just use once
# hierarchy=k_shell(G)
# print(hierarchy)
# svs = S_V(G, hierarchy)  # notice:k_shell will delete all the nodes
# print(svs)
# print(S_V_ba(G, hierarchy))



# print(ECRM(G, hierarchy))




