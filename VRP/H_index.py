import time
import matplotlib.pyplot as plp
import networkx as nx
# import igraph as ig
#
# import edgelist2gml
# import net2gml


# def HIndexJudgement(G, n):
#     '''
#     @author:
#         JianglongHu@github.com
#     @params:
#         graphFile: full name of input graph;
#     @return:
#         calcTime: time for calculation;
#         result: n-order H Index values of graph, type: List;
#     '''
#     # graphFileName, graphFileFormat = graphFile.split(".", 1)
#     #
#     # if graphFileFormat == 'edgelist':
#     #     edgelist2gml.edgelist2gml(graphFile)
#     # elif graphFileFormat == 'net':
#     #     net2gml.net2gml(graphFile)
#     #
#     # nxG = nx.read_gml(graphFileName + '.gml')
#     # igG = ig.Graph.Read_GML(graphFileName + '.gml')
#
#     start_time = time.clock()
#     # demo: result = calcH0IndexValues(nxG,igG)
#     # your solution:
#     result = calcHIndexValues(nxG, n)
#     end_time = time.clock()
#     calcTime = end_time - start_time
#
#     return (result, calcTime)


def calcH0IndexValues(nxG):
    result = [nxG.degree(v) for v in nx.nodes(nxG)]
    return result


# 计算h_index
def count_h_index(h_list):
    # 降序排序
    h_list = sorted(h_list, reverse=True)
    _len = len(h_list)
    i = 0
    while (i < _len and h_list[i] > i):
        i += 1
    return i


def cal_h_index(G, n, h_neg_dic):
    assert n >= 0, 'n>=0'  # 保证n>=0,否则报错
    # 0阶
    if n == 0:
        h_index_dic = {}  # 每个节点的0阶h指数
        for n_i in nx.nodes(G):
            h_index_dic[n_i] = nx.degree(G, n_i)
        return h_index_dic
    else:
        h_index_dic = {}
        n = n - 1
        h0_index_dic = cal_h_index(G, n, h_neg_dic)
        # print(n,h0_index_dic)
        for n_i in nx.nodes(G):
            h_list = []
            for neg in h_neg_dic[n_i]:
                h_list.append(h0_index_dic[neg])
            h_index_dic[n_i] = count_h_index(h_list)
        return h_index_dic


def calcHIndexValues(nxG, n):  # n=0是一阶
    h_neg_dic = {}
    for n_i in nx.nodes(nxG):
        a = []
        for neg in nx.neighbors(nxG, n_i):
            a.append(neg)
        h_neg_dic[n_i] = a
    result_dic = cal_h_index(nxG, n, h_neg_dic)
    result = []
    for val in result_dic.values():
        result.append(val)
    return result
# G = nx.Graph()
# G.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (3, 4), (4, 7), (5, 6)])
#
# print(calcHIndexValues(G, 0))
# nx.draw_networkx(G)
# plp.show()