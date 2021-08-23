import networkx as nx
import math
import operator
import matplotlib.pyplot as plp

'''
MCDE(v) = core(v) + degree(v) + entropy(v)
pi = (v's neighbors occur in ith shell) / d_v
entropy = -sum_i=0^max(shell) pi*log2(po)
'''
def get_weight(G, degree_dic, rank):  # wij表示：i给j投票的能力，wij ！= wji, rank is seed noods
    weight = {}
    nodes = nx.nodes(G)
    rank_list = [i[0] for i in rank]
    for node in nodes:
        sum1 = 0
        neighbors = list(nx.neighbors(G, node))
        neighbors_common_rank = list(set(neighbors) & set(rank_list))
        if len(neighbors_common_rank) != 0:  # 节点与已选节点的直接为0
            for nc in neighbors_common_rank:
                weight[(node, nc)] = 0
        neighbours_without_rank = list(set(neighbors) - set(rank_list))  # voting for unselected nodes
        if len(neighbours_without_rank) != 0:  # if the node has other nieghbours
            for nbr in neighbours_without_rank:
                sum1 += degree_dic[nbr]
            for neigh in neighbours_without_rank:
                weight[(node, neigh)] = degree_dic[neigh] / sum1
        else:  # 当前节点只有已选节点作为邻居
            for neigh in neighbors:
                weight[(node, neigh)] = 0
    # for i, j in nx.edges(G):
    #     sum1 = 0
    #     for nbr in nx.neighbors(G, i):
    #         sum1 += degree_dic[nbr]
    #     weight[(i, j)] = degree_dic[j] / sum1
    return weight
def get_node_score(G, nodesNeedcalcu, node_ability, degree_dic, rank):

    weight = get_weight(G, degree_dic, rank)
    node_score = {}
    for node in nodesNeedcalcu:  # for ever node add the neighbor's weighted ability
        sum2 = 0
        for nbr in nx.neighbors(G, node):
            sum2 += node_ability[nbr] * weight[(nbr, node)]
        node_score[node] = sum2
    return node_score
def get_node_score2(G, nodesNeedcalcu, node_ability, degree_dic, rank):

    weight = get_weight(G, degree_dic, rank)
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
def get_keys(d, va):
    for k, v in d.items():
        if va in v:
            return k
def next_f(value, lis):
    '''
    :param value: the value needed to compare
    :param lis: list
    :return: 返回列表lis中第一个不等于value的元素下标，如果都等于，则默认返回第一个元素下标
    '''

    for i in lis:
        if i != value:
            return lis.index(i)
    return 0
def k_shell(graph):
    importance_dict = {}
    level = 1
    while len(graph.degree):
        importance_dict[level] = []
        while True:
            level_node_list = []
            for item in graph.degree:
                if item[1] <= level:
                    level_node_list.append(item[0])
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree):
                return importance_dict
            if min(graph.degree,key=lambda x: x[1])[1] > level:
                break
        level = min(graph.degree,key=lambda x:x[1])[1]
    return importance_dict
def voterank_plus(G, l, lambdaa):
    '''

    :param G: use new indicator + lambda + voterank, the vote ability = log(dij)
    :param l: the number of spreaders
    :param lambdaa: retard infactor
    :return:
    '''
    rank = []

    # count dict
    nodes = list(nx.nodes(G))
    degree_li = nx.degree(G)
    d_max = max([i[1] for i in degree_li])
    degree_dic = {}
    for i in degree_li:
        degree_dic[i[0]] = i[1]

    # node's vote information
    node_ability = {}
    for item in degree_li:
        degree = item[1]
        node_ability[item[0]] = math.log(1 + (degree/d_max))  # ln(x)

    # node_ability_values = node_ability.values()
    # degree_values = degree_dic.values()
    # weaky = mean_value(node_ability_values) / mean_value(degree_values)
    # node's score
    node_score = get_node_score2(G, nodes, node_ability, degree_dic, rank)



    for i in range(l):
        # choose the max entropy node for the first time t aviod the error
        max_score_node, score = max(node_score.items(), key=lambda x: x[1])
        rank.append((max_score_node, score))
        # set the information quantity of selected nodes to 0
        node_ability[max_score_node] = 0
        # set entropy to 0

        node_score.pop(max_score_node)
        # for the max score node's neighbor conduct a neighbour ability surpassing
        cur_nbrs = list(nx.neighbors(G, rank[-1][0]))  # spreader's neighbour 1 th neighbors
        next_cur_neigh = []  # spreader's neighbour's neighbour 2 th neighbors
        for nbr in cur_nbrs:
            nnbr = nx.neighbors(G, nbr)
            next_cur_neigh.extend(nnbr)
            node_ability[nbr] *= lambdaa  # suppress the 1th neighbors' voting ability

        next_cur_neighs = list(set(next_cur_neigh))  # delete the spreaders and the 1th neighbors
        for ih in rank:
            if ih[0] in next_cur_neighs:
                next_cur_neighs.remove(ih[0])
        for i in cur_nbrs:
            if i in next_cur_neighs:
                next_cur_neighs.remove(i)

        for nnbr in next_cur_neighs:
            node_ability[nnbr] *= (lambdaa ** 0.5)  # suppress 2_th neighbors' voting ability
        # find the neighbor and neighbor's neighbor
        H = []
        H.extend(cur_nbrs)
        H.extend(next_cur_neighs)
        for nbr in next_cur_neighs:
            nbrs = nx.neighbors(G, nbr)
            H.extend(nbrs)

        H = list(set(H))
        for ih in rank:
            if ih[0] in H:
                H.remove(ih[0])
        new_nodeScore = get_node_score2(G, H, node_ability, degree_dic, rank)
        node_score.update(new_nodeScore)

        # choose the max entropy node
        # sorted_score_li = sorted(node_score.items(), key=operator.itemgetter(1), reverse=True)
        #
        #
        # late_max_shell = get_keys(k_shell_re, rank[-1][0])
        # shell_set = []
        #
        # for i in sorted_score_li:
        #     shell_set.append(get_keys(k_shell_re, i[0]))  # corresponding shell value in sorted list[node_shell, node_shell]
        # shell =next_f(late_max_shell, shell_set)



        # node = sorted_score_li[shell]
        # rank.append(node)
        # node_ability[node[0]] = 0
        # node_score.pop(node[0])


        # while len(sorted_score_li) != 0:
        #     if len(set(shell_set)) != 1:  # 存在不同shell的节点
        #         max_score_node = sorted_score_li[index][0]
        #         score = sorted_score_li[index][1]
        #         curr_max_shell = shell_set[index]
        #         if curr_max_shell != late_max_shell:  # 这轮选的点和上轮不在一个shell里
        #             rank.append((max_score_node, score))
        #             node_ability[max_score_node] = 0
        #             node_score.pop(max_score_node)
        #             break
        #         else:
        #             index += 1
        # else:
        #     max_score_node = sorted_score_li[index][0]
        #     score = sorted_score_li[index][1]
        #     rank.append((max_score_node, score))
        #     node_ability[max_score_node] = 0
        #     node_score.pop(max_score_node)


        # node_score.pop(max_score_node)

        # #set the information quantity of selected nodes to 0
        # node_ability[max_score_node] = 0
        # # set entropy to 0
        # node_score.pop(max_score_node)
        # print(i, rank)
    return rank

# G = nx.read_edgelist('CEnew.edgelist', nodetype=int)
# G = nx.Graph()
# G.add_nodes_from(list(range(1, 27)))
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 11), (2, 12), (3, 4), (3, 8), (4, 23), (5, 9), (5, 10), (8, 26), (8, 25), (13, 15), (14, 15), (15, 17), (17, 23), (16, 23), (18, 23), (19, 23), (21, 23), (22, 23), (20, 23), (24, 25)])
# nx.draw_networkx(G)
# plp.show()
# print(voterank_plus(G, 9, 0.1))
#
# import secondMethod
# print(secondMethod.newMethod(G, 7))
# print(voterank_plus(G,15,0.1))
# G = nx.read_edgelist('power.edgelist')
# nx.draw_networkx(G)
# plp.show()