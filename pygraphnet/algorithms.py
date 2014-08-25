import graph
import json
import threading
import functools
from itertools import *
from collections import defaultdict, deque
from queue import Queue, LifoQueue
from heapq import heappush, heappop, heapify
from math import pow
import math
import random
import graph
from abstract import ShortestPath, Codes
import numpy as np

#Write info for the each step
#http://www.columbia.edu/~mc2775/publications.html
#http://en.wikipedia.org/wiki/Book:Graph_Algorithms
#http://en.wikipedia.org/wiki/K_shortest_path_routing

#Поработать над алгоритмами с циклическими графами

#Course 6.889: Algorithms for Planner Graphs and Beoynd (Fall 2011)

#Бенчмаркинг кода http://www.huyng.com/posts/python-performance-analysis/
#Просмотр объектов  виде графа http://mg.pov.lt/objgraph/

#Structure of complex network (стр 18)
#http://www.maths.qmul.ac.uk/~latora/report_06.pdf

#Много информации по графам
#http://www.math.cmu.edu/~ctsourak/amazing.html

#Анализ кода
#http://www.pylint.org/

class GraphAlgorithms:
    @classmethod
    def easy_short_path(self, graph, start, end):
        visited = set()
        path = []
        for node in graph.edges():
            if node.node not in visited:
                visited.add(node.node)
                if node.node == end:
                    return None
            value = node.get_edges()
            minnode = min(value,
                          key=lambda x: x.weight)
            path.append(minnode.inedge)
        return path

    #Breadth-First Search
    '''start - the root node
       graph - current graph
       multithread - enable multithread way for this algorithm (not yet)
    '''
    def bsf(self, graph, start, end, multithread=False):
        return BreadthFirstSearch(graph).run()


class BreadthFirstSearch(ShortestPath):
    def __init__(self, graph):
        super(BreadthFirstSearch, self).__init__(graph)
        self.path = {}

    def run(self, start, end):
        que = Queue()
        que.put(start)
        while que:
            v = que.get()
            for node in self.graph.get(v, []):
                self.path[node] = v
                que.put(node)
            if end == v:
                return self._final_path(start, [end])

    #http://www.eis.hu.edu.jo/deanshipfiles/pub1049316.pdf heuristic BFS
    def heuristric(self, target):
        leftq = Queue()
        rightq = Queue()

    def _final_path(self, start, result):
        if result[-1] == start:
            result.reverse()
            return result
        result.append(self.path[result[-1]])
        return self._final_path(start, result)


class DepthFirstSearch(ShortestPath):
    def __init__(self, graph):
        super(DepthFirstSearch, self).__init__(graph)
        self.visited = set()
        self.result = []

    def run(self, start):
        self.result.append(start)
        self.visited.add(start)
        for node in self.graph:
            if node not in self.visited:
                self.run(node)
        return self.result


class PruferCode(Codes):
    def __init__(self, graph):
        super(PruferCode, self).__init__(graph)

    def tocode(self):
        if self.graph.size() < 2:
            raise ValueError("Length of Graph is too short")

        knowgraph = self.graph
        prqueue = []
        while len(knowgraph.get_graph()) > 2:
            minvalue = filterfalse(lambda x: self.graph.adjacent(x),
                                       self.graph.get_graph())
            print(list(minvalue))
            knowgraph.delete_node(minvalue)
            heappush(prqueue, minvalue)

        return prqueue



#http://www.aaai.org/Papers/AAAI/2002/AAAI02-072.pdf
class DStar:
	def __init__(self, start, goal, data):
		self.start = start
		self.goal = goal
		self.data = data
		self.stack=[]
		self.stack.append(start)


	def _calc_key(self, key):
		pass

	def _distance(self, key):
		pass

	def _rhs(self, key):
		if key == self.start:
			return 0
		return min(self.distance(key)) #Доделать

	def _shortest_path(self):
		if len(self.start) == 0:
			raise Exception("This stack is empty")
		while self.start[0] < self._calc_key(self.goal):
			value = self.start.pop()
			if self._distance(value) > self.rhs(value):
				pass

	def loop(self):
		while True:
			path = self._shortest_path()

	def _update(self):
		pass

'''sgraph
graph of nodes with edges'''
class Prim:
    #graph = Tree in this case
    def __init__(self, sgraph):
        self.sgraph = sgraph
        self.prev=[None]*self.sgraph.size()
        self.key=[float("inf")]*self.sgraph.size()
        self.nodes = sgraph.nodes()

    #Обязательно нужно сделать с очередью с приоритетом
    def run(self,root):
        stree=[]
        used=[]
        spanning_tree={}
        stree.append(self.sgraph.nodes()[0].node)
        used.append(self.sgraph.nodes()[0])
        #heapify(stree)
        #heappush(stree, root)
        while len(used) > 0:
            node = used.pop()
            minnode = None
            minweight = 1000000
            for cnode in node.connected:
                if cnode.outedge[0].node not in stree\
                and cnode.weight < minweight:
                    minweight = cnode.weight
                    minnode = cnode.outedge[0]
            if minnode !=  None:
                print(minnode)
                used.append(minnode)
                stree.append(minnode.node)
            for othernode in self.sgraph.nodes():
                if othernode.node not in stree:
                    used.append(othernode)
                    stree.append(othernode.node)
                    break
        return stree



    def cheapestway(self, node):
        #Рекурсия
        print(min(node.connected,key=lambda x:x.weight).outedge)
        return min(node.connected,key=lambda x:x.weight).outedge[0]
        #return min(node.connected,key=lambda x:x.weight)

'''Алгоритмы планирования пути типа A*
http://cstheory.stackexchange.com/questions/11855/how-do-the-state-of-the-art-pathfinding-algorithms-for-changing-graphs-d-d-l '''

#Kruskal algorithm for minimum spanning tree
class Kruskal:
    def __init__(self, sgraph):
        self.sgraph = sgraph

    def run(self, root):
        newtree=[]
        used=[]
        spanning_tree={}
        if self.sgraph.size() == 0 :
            raise "Graph is empty"
        newtree.append(self.sgraph.nodes()[0].node)
        used.append(self.sgraph.nodes()[0])
        #print(self.sgraph.nodes()[0].connected[0])
        while len(used) > 0:
            popnode = used.pop()
            minvalue = 10000
            minnode = None
            for cnode in popnode.connected:
                out_node = cnode.outedge[0]
                if cnode.weight < minvalue and\
                 out_node!= None and out_node.node not in newtree:
                    minvalue = cnode.weight
                    minnode =cnode
            if minnode == None:break
            if minnode.outedge[0].node not in newtree:
                used.append(minnode.outedge[0])
                newtree.append(minnode.inedge.node)
            else:
                break
            spanning_tree[minnode.inedge.node] = minnode.outedge[0].node

        print(spanning_tree)


#http://courses.csail.mit.edu/6.006/fall11/rec/rec19.pdf
#http://habrahabr.ru/post/190850/ отсечение по ответу
class Dijkestra:
    def __init__(self, sgraph):
        self.sgraph = sgraph
        self.djk_used_nodes=[]

    #Goal find distance (and shortest path) from s(root) to every other vertex
    def run(self, root):
        if self.sgraph.size() == 0:
            raise "Graph is Empty"
        if self.sgraph.has_node(root):
            min_node = min(self.sgraph[root].connected, 
                key=lambda x:x.weight and x.inedge not in self.djk_used_nodes)
            self.djk_used_nodes.append(min_node.inedge)

    def runs(self, root):
        distance=[]
        heap =[]
        infinity = 1000000
        visited={}
        weights = {}
        heappush(heap, root)
       #print(self.sgraph[h].connected)
        while len(heap) > 0:
            extr_node= heappop(heap)
            visited[extr_node] = True
            extr_node = self.sgraph[extr_node]
            #print(extr_node)
            for node in extr_node.connected:
                current_node = node.outedge[0].node
                if current_node not in visited:
                    weights[current_node] = min(weights[current_node], \
                        weights[extr_node] + current_node.weight)
                    heappush(heap, current_node)

    def _extract_min(self, list_of_nodes):
        return min(
            map(lambda x:(x.weight, x.value), 
            list_of_nodes))
            





"""Find goal; edge and return all nodes between
#this edge
example:
{"A":"B", label=current"} =>
FindLabel("current) -> ["A","B"]
"""

"Path finding algorithm"
class FindLabel:
    def __init__(self, sgraph):
        self.sgraph = sgraph

    def run(self, edge):
        marks = []
        temps=[]
        heappush(marks, edge)
        resultnodes=[]

        while len(marks) > 0:
            current = marks.pop()
            for node in current.connected:
                if node.label == edge:
                    resultnodes.append(node.connected)
                    return resultnodes
                if node not in temp:
                    marks.append(node)
                    temp.append(node)



#Max flow problem

class Algorithms:
    def __init__(self, sgraph):
        #make cache for immutable objects
        self.cache = {}
        self.sgraph = sgraph

    def BSF(self, target):
        if(target in self.cache):
            return self.cache[target]
        b = BreadthFirstSearch()
        result = b.run(target)
        self.cache[target] = result
        return result

    def DFS(self, target):
        if target in self.cache[target]:
            return self.cache[target]
        d = DepthFirstSearch()
        result = d.run(target)
        self.cache[target] = result
        return result

    #http://en.wikipedia.org/wiki/Bridge_%28graph_theory%29#Bridge-finding_algorithm
    #e-maxx стр 93
    def find_bridge(self):
        timer = 0
        used=[False] * self.sgraph.size()
        for node in range(self.sgraph.size()):
            if used[node] != False:
                DepthFirstSearch(self.sgraph).run(node)

    "Most popular path on graph"
    def most_popular_path(self):
        path=[]
        path.append(self.sgraph.nodes()[0])
        while len(path) > 0:
            current_node = path.pop()
            pop_node=None
            for node in self.sgraph[current_node]:
                if(node.weight > current_node.weight):
                    pop_node = node
            if pop_node != None:
                path.append(pop_node)


    #clustering with random walk
    #http://www.umiacs.umd.edu/~mingyliu/enee698a/hara.pdf
    def random_walk(self):
        for node in self.sgraph.nodes():
            if node.connected != []:
                lens = 100/len(node.connected)
                node.walk_probability = lens


    ''' динамический алгоритм для нахождения 
    кратчайших расстояний между всеми вершинами взвешенного ориентированного графа.
    W - matrix'''
    def Floyd_Warshall(self, W):
        for i in range(W.size()):
        	for j in range(W.size()):
        		for k in range(W.size()):
        			W[i][j] = min(W[i][j], W[i][k] + W[k][j])


    #http://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
    def BelmannFord(self):
        '''Алгоритм Белмана-Форда - поиск кратчайшего пути
        во взвешанном графе. Допускает рёбра с отрицательным весом
        '''
        w = None #some weight
        key_length = len(self.sgraph.keys())
        #Строим матрицу для нахождения кр пути
        arr = [np.zeros(key_length) for i in range(key_length)]
        #-1 вершина
        for node,i in enumerate(self.sgraph.keys()):
            for jnode,j in enumerate(self.sgraph[node].connected):
                check = arr[i][j-1] + gr.edge(node, jnode).weight
                if arr[i][jnode] > check:
                    arr[i][jnode] = check
        #Check negative weight cycles
        for node, i in enumerate(self.sgraph.keys()):
            if dist(node) + w < dist(node):
                print("Graph contains negative cycles")
                break
            

    '''Алгоритм Джонсона
    Использование Belmann-Ford алгоритма'''
    def JohnsonAlgorithm(self):
        pass

    #D star-lite http://pub1.willowgarage.com/~konolige/cs225b/dlite_tro05.pdf
    def DStar(self, data, start, goal):
    	#Compute shortest path
    	pass

    '''Метод Шульце
    http://en.wikipedia.org/wiki/Schulze_method
    matrxgraph - граф в виде матрицы (numpy matrix type)

       a   b   c
    a   -   10  20

    b   15   -  15
 
    c   30  10   -
    '''
    def Schulze_voiting(self, matrxgraph):
        size = len(matrxgraph)
        newarr = np.array(np.zeros((size,size)), dtype=int)
        for X in range(len(matrxgraph)):
            for Y in range(len(matrxgraph)):
                if X != Y:
                    if matrxgraph[X,Y] > matrxgraph[Y,X]:
                        newarr[X,Y] = matrxgraph[X,Y]

        return self._Schulze_voiting_compute(matrxgraph, newarr, size)

    def _Schulze_voiting_compute(self, matrxgraph, resmatrix, size):
        for X in range(len(matrxgraph)):
            for Y in range(len(matrxgraph)):
                if X != Y:
                    for C in range(len(matrxgraph)):
                        if X != C and Y != C:
                            resmatrix[Y,C] = max(resmatrix[Y,C], 
                                min(resmatrix[Y,X], resmatrix[X,C]))
        return resmatrix

    def sch_voiting_alt(self, matrxgraph):
        print(list(
            filter(lambda X: 
            (lambda Y: len(Y), X),
             matrxgraph)))


    def dijkestra(self, target):
        return Dijkestra(self.sgraph).runs(target)

class Eurelian:
    """docstring for EurelianPath"""
    def __init__(self, egraph):
        self.egraph = egraph
        self.path = []

    def is_path(self, egraph):
        nodes = egraph.nodes()
        return nodes == self._walk_in(path)

    def path(self, egraph):
        return self._walk_in(egraph)

    #Эйлеров цикл
    def cycle(self, egraph):
        cycle=[]
        notvisited = list(adlist.keys())
        m = 0
        while notvisited != []:
            newStart = choice(notvisited)
            notvisited.remove(newStart)
            nv = list(adlist.keys())
            ads = {}
            arr = []
            visited=[]
            visited.append(newStart)
            edges={}
            while visited != []:
                newStart = visited[::-1][0]
                newEdge = None
                for x in adlist[newStart]:
                    if (newStart, x) not in edges:
                        newEdge = (newStart, x)
                        break
                if newEdge != None:
                    edges[(newStart,x)] = 0
                    visited.append(x)
                    arr.append(x)
                else:
                    value = visited.pop()
            if len(arr) > m:
                m = len(arr)
                cycle = arr
        return cycle

    def _walk_in(self, egraph):
        nodes = egraph.nodes()
        numqueue=[]
        heapify(numqueue)
        heappush(numqueue, 'a')
        for node in nodes:
            entries = node.connected
            if len(entries) > 0:
                heappush(entries[0])


#Алгоритм Левита Нахождение кратчайшего расстояния от одной вершин графа до всех остальных
#Or in Englisg SPFA
class Levit:
    def __init__(self, sgraph):
        self.sgraph = sgraph
        self.completeNodes={} #Уже вычисленные вершины (бесконечные значения)
        self.inprocessNodes={} #В процессе вычисления
        self.notComplete = {} #Ещё не вычислено

    #Расчёт нодов
    def fit(self, node):
        if node in self.notComplete:
            self.notComplete[node] = 0
            self.notComplete = {node:self.graph.size() + 1 for node in self.sgraph.items()}
            i = self.sgraph.size()
            q = deque()
            q.append(node)
            while not q.isEmpty():
                current_node = q.pop()
                for child_node in self.connected:
                    length = child_node.weight  #length or weight
                    name = child_node.name
                    if child_node in self.notComplete:
                        del self.notComplete[child_node]
                    if child_node in self.inprocessNodes:
                        self.inprocessNodes[child_node] = current_node.weight + child_node.weight
                    if child_node in self.completeNodes:
                        if self.completeNodes[child_node] > self.self.completeNodes[current_node] + child_node.weight:
                            q.append(child_node)




'''Most popular nodes'''

#Добавить алгоритм Graph Matching Monte-Carlo
#http://cv.snu.ac.kr/publication/conf/2010/GMDDMCMC_ICPR2010.pdf

#Изоморфизм и прочие морфизмы графовграфов
#http://www.dharwadker.org/tevet/isomorphism/

class GraphMorph:
    def __init__(self, sgraph):
        self.sgraph = sgraph

    #http://math.stackexchange.com/questions/393416/are-these-2-graphs-isomorphic
    #http://en.wikipedia.org/wiki/Graph_isomorphism_problem

    '''isomorphic graphs defenition
           Two graphs which contain the same number 
          of graph vertices connected in the same way are said to be isomorphic. (Wolfram Mathworld)
          http://mathworld.wolfram.com/IsomorphicGraphs.html
    ''' 

    '''TO DO вычислить одинаковую связность Canonical labeling'''
    '''Practical graph isomorp.. http://arxiv.org/pdf/1301.1493v1.pdf'''
    def is_isomorphic(self, another_graph):
        if self.sgraph.size() != another_graph.size():
            print ('Those graphs is not isomorphic')

        #Вершины и количество связей между ними {Вершина, количество связей}
        nodes_from_first = list(self.sgraph.connected_counts())
        nodes_from_second = list(another_graph.connected_counts())
        length_first_graph = len(list(self.sgraph.connected_counts()))
        length_second_graph = len(list(another_graph.connected_counts()))
        if length_first_graph != length_second_graph:
            print ('Those graphs is not isomorphic')

        #O^2 in worst case
        for node1 in nodes_from_first:
            current_node = list(node1)[1]
            find_identical_connected = False
            for node2 in nodes_from_second:
                check_node = list(node2)[1]
                if len(current_node.connected) == len(check_node.connected): 
                    find_identical_connected = True
            if not find_identical_connected:
                print ('Graphs is not isomorphic')
                return False
        return True

    def _create_tuple_of_graphs():
        nt = namedtuple('iso', 'graph1 graph2')



#Basic algorithms for min cut problem
class MinimumCut:
    def __init__(self, graph):
        self.egraph = egraph

    #http://en.wikipedia.org/wiki/Karger%27s_algorithm
    def Karger(self, start, iters=1000):
        lnodes=[]
        #Псевдокод
        for i in range(iters):
            nodes = self.egraph.nodes()
            while len(nodes) > 2:
                ch =choice(self.egraph.edges())
                target = ch.innode
                merge(ch.innode, outnode)
                del nodes[target]
            lnodes.append(nodes)
        #Return minimum frequence of nodes
        return sorted(lnodes, lambda x: x)

#Minimum k-cut problem
#http://math.mit.edu/~goemans/18434S06/multicuts-brian.pdf
#find the minimum-weight set of edges
class MinimumKcuts:
    def __init__(self, egraph):
        '''
            Need egraph with minimum cut
        '''
        self.egraph = egraph

    #http://arxiv.org/pdf/1310.0178v1.pdf
    def gomoryHuTree(self):
        nodes = self.egraph.nodes()
        newgraph = Graph()
        edges = []
        if len(nodes) < 2:
            return 0
        node = choice(nodes)

##Check Philip Kelin pubs about graphs
#Graph minor
#H - undirected graph is called a minor of the graph G if H can be
#formed from G by deleting edges and vertices and by contracting edges

#also check matroid minors
#Полиномиальное время!
#Check approximate distance

#from bilder import PlanarGraph
class GraphMinor:
    def __init__(self, sgraph):
        self.sgraph = sgraph

    def check_monor(self, hgraph):
        pass


#Интерфейс для разных видов случайных графов
class PuppetRandomGraph:
    def generate(self):
        pass

#A Sequential algorithm for Generating Random Graphs (2009)
#seq - degree sequence
#n - number of iteration
class SequenceRandomGraph(PuppetRandomGraph):
    def __init__(self, seq, n):
        self.seq = seq
        self.n = n

    def _emptyGraph(self):
        return graph.Graph()

    def generate(self):

        #Начинаем с пустого графа
        #ПОследовательно добавляем связи между парами несвязных вершин
        g = self._emptyGraph()
        connect = lambda: random.choice(self.seq)
        list(map(lambda x: g.add_edge(connect(), connect()), range(self.n)))
        return g

    def generate_advanced(self):
        g = self._emptyGraph()


#http://www.iecn.u-nancy.fr/~chassain/djvu/SpencerStFlour.pdf
class RandomGraph(PuppetRandomGraph):
    def __init__(nodes):
        self.nodes = nodes
        self.length = len(nodes)

    def generate(self):
        raise NotImplemented


#Генерируем случайную связанность с помощью распределений из np.random
#Например, gamma distribution
class RandomGraphDistribution(PuppetRandomGraph):
    def __init__(self, nodes):
        super(RandomGraphDistribution, self).__init__()
        self.nodes = nodes
        self.length = len(nodes)
        self.sgraph = graph.Graph()

    def generate(self):
        nnodes = [round(x) for x in np.random.gamma(1, 5, self.length)]
        #self.sgraph.add_nodes(nnodes)
        nnodes2 = [round(x) for x in np.random.gamma(2, 2, self.length)]
        for innode, outnode in zip(nnodes, nnodes2):
            self.sgraph.add_edge(innode, outnode)
        #Проверить правильность добавления!

        #print('Distribution: ', nnodes, nnodes2)
        #print(self.nodes)

def test_nodes():
    g = graph.Graph()
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_node(5)
    '''g.add_node(6)
    g.add_node(7)
    g.add_node(8)'''
    #g.add_edge('a')
    g.add_edge(1,3,weight=7)
    g.add_edge(1,4,weight=4)
    g.add_edge(1,6,weight=3)
    g.add_edge(2,3,weight=2)
    g.add_edge(3,4,weight=3)
    g.add_edge(3,6,weight=6)
    g.add_edge(4,5,weight=1)
    g.add_edge(5,6,weight=3)
    g.add_edge(5,7,weight=2)
    g.add_edge(6,7,weight=4)
    g.add_edge(7,1,weight=1)
    g.add_edge(7,8,weight=4)
    g.add_edge(8,1,weight=2)
#g.add_edge_random(4,max_weight=50)

def test_kruskal():
    g = graph.Graph()
    '''g.add_node(1)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_node(5)'''
    g.add_nodes([], start = 1, step=1, end=5 )
    g.add_nodes([], r=[1,1,5])
    g.add_edge(1,4, weight=1, rev=True)
    g.add_edge(1,5, weight=3, rev=True)
    g.add_edge(4,5, weight=2, rev=True)
    g.add_edge(4,2, weight=5, rev=True)
    g.add_edge(5,2, weight=4, rev=True)
    g.add_edge(5,3, weight=4, rev=True)
    g.add_edge(2,3, weight=3, rev=True)
    pr = Kruskal(g)
    pr.run(1)
    #http://www.frc.ri.cmu.edu/~axs/doc/icra94.pdf

def test_prim():
    g = graph.Graph()
    g.add_node(1)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_node(5)
    g.add_edge(1,4, weight=1, rev=True)
    g.add_edge(1,5, weight=3, rev=True)
    g.add_edge(4,5, weight=2, rev=True)
    g.add_edge(4,2, weight=5, rev=True)
    g.add_edge(5,2, weight=4, rev=True)
    g.add_edge(5,3, weight=4, rev=True)
    g.add_edge(2,3, weight=3, rev=True)
    pr = Prim(g)
    print(pr.run(2))


#Better MST. Tree contraction
'''
Tree contraction
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/phrensy/pub/papers/LeisersonM88/node15.html
http://www.keisu.t.u-tokyo.ac.jp/research/techrep/data/2008/METR08-27.pdf Non B trees
http://www.cs.duke.edu/~reif/paper/tate/dyntree/dyntree.pdf Описание
На вход подаётся бинарное дерево

На первом этапе, вершины A и B сливаются (Merge), также как и ноды C D
Новое дерево формируется из вершин E и F и соединяются  с AB и CD. Дальше,
E соединяется с CD и образует ноду CDE. После нескольких шагов, все вершины сливаются
в единую ноду

1. Делаем пару из входных узлов
2. Следующая пара выбирается из соседей. Узел с двумя детьми выбирает потомка с вероятностью 1/2
3. Если два узла подбирают друг друга(видимо случайным образом), то они сливаются
Алгоритм O(lg n)

'''

def tree_contraction(bin_tree):
    splee=[]
    used=[]
    used.append(bin_tree[0])
    while len(used) > 0:
        node = used.pop()
        for new_node in node.values():
            splee.append([node, newtree])
    return splee


def construct_graph():
    g = graph.Graph({0:[3], 1:[2], 2:[3], 3:[0, 1]})
    g.add_node(1)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_node(5)
    g.add_node(6)
    g.add_edge(1,4, weight=5)
    g.add_edge(1,2, weight=6)
    g.add_edge(1,3, weight=1)
    g.add_edge(2,3, weight=5)
    g.add_edge(2,5, weight=3)
    g.add_edge(5,3, weight=6)
    g.add_edge(5,6, weight=6)
    g.add_edge(6,3, weight=4)
    g.add_edge(6,4, weight=2)
    g.add_edge(5,6, weight=6)
    g.add_edge(4,3, weight=5)
    return g


def test_random_walk():
    gr = construct_graph()
    algo = Algorithms(gr)
    algo.random_walk()

def test_prim2():
    pr = Prim(construct_graph())
    print(pr.run(2))
#http://www.frc.ri.cmu.edu/~axs/doc/icra94.pdf
class Dstar:
    def __init__(self):
        pass


def test_eurelian():
    path = {0:[3], 1:[2], 2:[3], 3:[0, 1]}
    gr = graph.Graph(path)
    eu = Eurelian(gr)
    eu.path(gr)

def optional_test_rank():
    gr = construct_graph()
    #print(gr.set_simple_rank(4, lambda x:x.weight > 2))

def test_morph():
    gr = construct_graph()
    morph = GraphMorph(gr)
    morph.is_isomorphic(gr)

#Алгоритм Дейкестры вики
'''BUG with multi connection !!!!!!
g.add_edge(6,3, weight=4, rev=True)
g.add_edge(3,6, weight=4, rev=True)
'''
def dijkestra_test():
    path = {1:[2,3,6], 2:[1,4,3], 3:[1,2,4,6], 4:[3,2,5], 5:[4,6], 6:[5,3,1]}
    gr = graph.Graph(path)
    gr.add_edge(1,2, weight=7)
    gr.add_edge(1,3, weight=9)
    gr.add_edge(1,6, weight=14)
    gr.add_edge(2,1, weight=7)
    gr.add_edge(2,4, weight=15)
    gr.add_edge(2,3, weight=10)
    gr.add_edge(3,1, weight=9)
    gr.add_edge(3,6, weight=2)
    gr.add_edge(3,4, weight=11)
    gr.add_edge(3,2, weight=10)
    gr.add_edge(4,3, weight=11)
    gr.add_edge(4,2, weight=15)
    gr.add_edge(4,5, weight=6)
    gr.add_edge(5,6, weight=9)
    gr.add_edge(5,4, weight=6)
    gr.add_edge(6,5, weight=9)
    gr.add_edge(6,1, weight=14)
    gr.add_edge(6,3, weight=2)
    al = Algorithms(gr)
    al.dijkestra(1)


def test_sequence_random_graph():
    s = SequenceRandomGraph([5,2,3,1], 4)
    s.generate()

def test_random_graph_distibution():
    r = RandomGraphDistribution([2,5,7,8,6,9])
    r.generate()

test_random_graph_distibution()
#test_sequence_random_graph()
#dijkestra_test()
#test_morph()
#test_random_walk()

