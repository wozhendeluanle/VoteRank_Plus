# -*- coding: utf-8 -*-
"""
Implement linear threshold models
社交网络影响力最大化 传播模型——线性阈值（LT）模型算法实现
"""
import copy
import itertools
import random
import math
import networkx as nx

__all__ = ['linear_threshold']

#-------------------------------------------------------------------------
#  Some Famous Diffusion Models
#-------------------------------------------------------------------------

def linear_threshold(G, seeds, steps=0):           #LT线性阈值算法
  """
  Parameters
  ----------
  G : networkx graph                     #所有节点构成的图
      The number of nodes.

  seeds: list of nodes                   #子节点集
      The seed nodes of the graph

  steps: int                             #激活节点的层数（深度），当steps<=0时，返回子节点集能激活的所有节点
      The number of steps to diffuse
      When steps <= 0, the model diffuses until no more nodes
      can be activated

  Return
  ------
  layer_i_nodes : list of list of activated nodes
    layer_i_nodes[0]: the seeds                  #子节点集
    layer_i_nodes[k]: the nodes activated at the kth diffusion step   #该子节点集激活的节点集

  Notes
  -----
  1. Each node is supposed to have an attribute "threshold".  If not, the
     default value is given (0.5).    #每个节点有一个阈值，这里默认阈值为：0.5
  2. Each edge is supposed to have an attribute "influence".  If not, the
     default value is given (1/in_degree)  #每个边有一个权重值，这里默认为：1/入度

  References
  ----------
  [1] GranovetterMark. Threshold models of collective behavior.
      The American journal of sociology, 1978.
  """

  if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
      raise Exception( \
          "linear_threshold() is not defined for graphs with multiedges.")

  # make sure the seeds are in the graph
  for s in seeds:
    if s not in G.nodes():
      raise Exception("seed", s, "is not in graph")

  # change to directed graph
  if not G.is_directed():
    DG = G.to_directed()
  else:
    DG = copy.deepcopy(G)        # copy.deepcopy 深拷贝 拷贝对象及其子对象

  # init thresholds
  for n in DG.nodes():  # for each node
    if 'threshold' not in DG.nodes[n]:
      DG.nodes[n]['threshold'] = 0.25
    elif DG.nodes[n]['threshold'] > 1:
      raise Exception("node threshold:", DG.nodes[n]['threshold'], \
          "cannot be larger than 1")

  # init influences
  in_deg = DG.in_degree()       #获取所有节点的入度
  for e in DG.edges():
    if 'influence' not in DG[e[0]][e[1]]:
      DG[e[0]][e[1]]['influence'] = 1.0 / in_deg[e[1]]    #计算边的权重
    elif DG[e[0]][e[1]]['influence'] > 1:
      raise Exception("edge influence:", DG[e[0]][e[1]]['influence'], \
          "cannot be larger than 1")

  # perform diffusion
  A = copy.deepcopy(seeds)
  if steps <= 0:
    # perform diffusion until no more nodes can be activated
    return _diffuse_all(DG, A)
  # perform diffusion for at most "steps" rounds only
  return _diffuse_k_rounds(DG, A, steps)

def _diffuse_all(G, A):
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])
  while True:
    len_old = len(A)
    A, activated_nodes_of_this_round = _diffuse_one_round(G, A)
    layer_i_nodes.append(activated_nodes_of_this_round)
    if len(A) == len_old:
      break
  return layer_i_nodes

def _diffuse_k_rounds(G, A, steps):
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])
  while steps > 0 and len(A) < len(G):
    len_old = len(A)
    A, activated_nodes_of_this_round = _diffuse_one_round(G, A)
    layer_i_nodes.append(activated_nodes_of_this_round)
    if len(A) == len_old:
      break
    steps -= 1
  return layer_i_nodes

def _diffuse_one_round(G, A):
  activated_nodes_of_this_round = set()
  for s in A:
    nbs = G.successors(s)
    for nb in nbs:
      if nb in A:
        continue
      set(G.predecessors(nb)).intersection(set(A))
      active_nb = list(set(G.predecessors(nb)).intersection(set(A)))
      if _influence_sum(G, active_nb, nb) >= G.nodes[nb]['threshold']:
        activated_nodes_of_this_round.add(nb)
  A.extend(list(activated_nodes_of_this_round))
  return A, list(activated_nodes_of_this_round)

def _influence_sum(G, froms, to):
  influence_sum = 0.0
  for f in froms:
    influence_sum += G[f][to]['influence']
  return influence_sum

def LT_Activited_nodes(LT_Result):
  # del LT_Result[0]  # don't include seed nodes
  sumOfActNodes = 0
  for i in LT_Result:
    sumOfActNodes += len(i)
  return sumOfActNodes
# G = nx.Graph()
# G.add_nodes_from(list(range(1, 27)))
# G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 11), (2, 12), (3, 4), (3, 8), (4, 23), (5, 9), (5, 10), (8, 26), (8, 25), (13, 15), (14, 15), (15, 17), (17, 23), (16, 23), (18, 23), (19, 23), (21, 23), (22, 23), (20, 23), (24, 25)])
# G = nx.read_edgelist('')
# seeds_origin = [(2, 3.0112161952256487), (23, 2.6654307630035396), (8, 1.1028714576364056), (15, 0.8795688030691878), (5, 0.5477382683502305), (25, 0.2729332014991936), (1, 0.038502685095507916), (3, 0.012735228433101065), (4, 0.0), (6, 0.0), (7, 0.0), (9, 0.0), (10, 0.0), (11, 0.0), (12, 0.0), (13, 0.0), (14, 0.0), (16, 0.0), (17, 0.0), (18, 0.0)]
# seeds = [i[0] for i in seeds_origin]
# results = linear_threshold(G, seeds[:4],steps=0)
# print(results)
# print('steps', len(results))
# print(LT_Activited_nodes(results))

