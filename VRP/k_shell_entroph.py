import networkx as nx
import H_index
import operator
import math
import matplotlib.pyplot as plp
# nx.draw_networkx(G)
# plp.show()


def originalinforEntro(G):
    '''
       原文中的信息熵函数，各个参数的意义：
       parameters:
           G: 图
      return:
           e_dic: 所有节点的信息熵字典 {node: information entrophy}
       '''
    nodes = list(nx.nodes(G))
    degree_dic = dict(nx.degree(G))

    I_dic = {}  # 对应原文中的I_i
    e_dic = {}  # 对应原文中的e_i
    sum_de = 0
    for j in nodes:
        sum_de += degree_dic[j]
    for i in nodes:
        I_dic[i] = degree_dic[i] / sum_de
    for k in nodes:
        e = 0
        for k_nei in nx.neighbors(G, k):
            e += I_dic[k_nei]*math.log(I_dic[k_nei])
        e_dic[k] = -e
    return e_dic
def k_shell(G):
    '''
    parameter:
        G :graph
    return:
        importance_dict: Kshell dict {ks : [node, ..., node]}
    '''
    graph = G.copy()
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
def dic_Sort(a):
    '''
    原文中排序方法的实现
    a: 字典
    '''
    sorted_a = sorted(a.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    b = []
    for i in range(len(a)):
        b.append(sorted_a[i][0])
    return b
def newmethod(G):
    k_shellsets = k_shell(G)
    I_dic = originalinforEntro(G)
    a = len(k_shellsets)
    sorted_kshellsets = dict(sorted(k_shellsets.items(), key=operator.itemgetter(0), reverse=True))  # big shell is in the front
    # print(sorted_kshellsets)
    partition = []  # the kshell partition with sorted information entrophy [{nodes:entrophy}]
    for k in sorted_kshellsets.keys():  # i in [3,2,1]
        # print(i)
        par_dic = {}
        for j in k_shellsets[k]:
            par_dic[j] = I_dic[j]
        partition.append(dic_Sort(par_dic))

    influencialNodes = []
    length = []
    for f in partition:
        length.append(len(f))  # get the length of each partition

    for maxL in range(max(length)):
        for k_shelll in range(len(partition)):  # k_shell = 0,1,2
            if maxL < len(partition[k_shelll]):  # 若现在的maxL小于目前列表的长度
                influencialNodes.append(partition[k_shelll][maxL])
            else:
                pass
    return influencialNodes



# G = nx.Graph()  # 原文中的示例图
# G.add_nodes_from(list(range(1, 27)))
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 11), (2, 12), (3, 4), (3, 8), (4, 23), (5, 9), (5, 10), (6, 7), (8, 26), (8, 25), (13, 15), (14, 15), (15, 17), (17, 23), (16, 23), (18, 23), (19, 23), (21, 23), (22, 23), (20, 23), (24, 25)])
G = nx.read_edgelist('E:/我的任务/Networks/experiments and data/social/Jazz.edgelist', nodetype=int)
print(newmethod(G))
# print(IKS(G))
# G = nx.read_edgelist('USAir97.edgelist', nodetype=int)
# G = nx.read_edgelist('football.edgelist', nodetype=int)
# G = nx.read_edgelist('jazz.edgelist', nodetype=int)

# nx.draw_networkx(G)
# plp.show()

# graph = ['amazon', 'condmat', 'dblp', 'email-Eu-core', 'Gowalla', 'HepPh', 'youtube']
# for name in graph:
#     G = nx.read_edgelist(name + '.edgelist', nodetype=int)
#     result = newmethod(G)
#     filename = name + '\'s spreaders.txt'
#     with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
#         f.write(str(result))

