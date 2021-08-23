import networkx as nx
import matplotlib.pyplot as plp
import math
import operator
'''
MCDE(v) = core(v) + degree(v) + entropy(v)
pi = (v's neighbors occur in ith shell) / d_v
entropy = -sum_i=0^max(shell) pi*log2(po)
'''
def get_keys(d, va):
    for k, v in d.items():
        if va in v:
            return k
def k_shell(graph):
    importance_dict={}
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
def MCDE(G):
    G1 = G.copy()
    nodes = list(nx.nodes(G))
    k_shell_re = k_shell(G1)
    degree_dic = {}
    for item in nx.degree(G):
        degree_dic[item[0]] = item[1]

    P = {}
    for node in nodes:
        p_li = [0] * len(k_shell_re)
        for nbr in list(nx.neighbors(G, node)):
            for i in range(len(k_shell_re)):
                shell = list(k_shell_re.values())
                if nbr in shell[i]:
                    p_li[i] += 1
        P[node] = [i / degree_dic[node] for i in p_li]


    entropy = {}
    for node in nodes:
        sum_entropy = 0
        for i in range(len(k_shell_re)):
            if P[node][i] == 0:
                sum_entropy += 0
            else:
                sum_entropy += (P[node][i] * math.log2(P[node][i]))
        entropy[node] = - sum_entropy

    MCDE = {}
    for node in nodes:
        MCDE[node] = nx.core_number(G)[node] + degree_dic[node] + entropy[node]
    sort_val_MCDE = sorted(MCDE.items(), key=operator.itemgetter(1), reverse=True)

    return sort_val_MCDE

# nx.draw_networkx(G)
# plp.show()
# print(k_shell(G1))

# G = nx.Graph()
# G.add_nodes_from(list(range(1, 24)))
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (1, 6), (1, 9), (1, 10), (2, 3), (2, 4), (2, 11), (2, 13), (2, 14), (2, 15),  (3, 4), (3, 7), (4, 7),  (4, 16), (4, 17), (4, 18), (5, 6), (5, 19), (5, 20), (7, 8), (7, 21), (11, 12), (16, 22), (16, 23)])
# #
# # G = nx.read_edgelist('jazz.edgelist')
# print(MCDE(G))







