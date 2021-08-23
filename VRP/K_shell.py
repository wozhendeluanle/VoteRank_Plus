import networkx as nx


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
# G = nx.read_edgelist('karate.edgelist')
# G = nx.Graph()
# G.add_nodes_from(list(range(1, 27)))
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 11), (2, 12), (3, 4), (3, 8), (4, 23), (5, 9), (5, 10), (6, 7), (8, 26), (8, 25), (13, 15), (14, 15), (15, 17), (17, 23), (16, 23), (18, 23), (19, 23), (21, 23), (22, 23), (20, 23), (24, 25)])
#
# print(k_shell(G))
