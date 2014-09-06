import json
import threading
import functools
from itertools import *
from collections import defaultdict
from queue import Queue, LifoQueue
from heapq import heappush, heappop, heapify
from math import pow
import math
import random
import unittest
import time
import re
import operator
from abstract import AbstractGraph
import json

import numpy as np

#Дуги между нодами, меняющиеся в зависимости от времени !
#Добавить кэширование результатов

#Помещать время добавления нода! (in StructNode - added_time)


#Загрузка графа
class LoadGraph:
    def __init__(self, fname):
        self.fname = fname
        self.graph = {}

    def load(self):
        for line in f.readlines():
            result = list(map(lambda x: int(x), line.split()))
            graph[result[0]] = result[1:]
        return graph

    #Reconstruct graph with adding another
    def __add__(self, fname):
        lg = LoadGraph(fname).load()
        for node in lg.keys():
            if node not in self.graph:
                self.graph[node] = lg[node]
            else:
                self.graph[node] += lg[node]


#В файле labs
class GraphDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        # self.graph = graph

    def __setitem__(self, node, value):
        super(GraphDict, self).__setitem__(node, value)


class EdgeDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __setitem__(self, node, edge):
        super(EdgeDict, self).__setitem__(node, edge)


class GraphError(Exception):
    def __init__(self, value):
        Exception.__init__(self)


class StructNode:
    def __init__(self, node, value, attribute=[], **kwargs):
        self.node = node
        self.value = value
        self.attribute = attribute
        self.index = kwargs.get('index', [])
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.edges = kwargs.get('edges')  # for multigraph
        self.weight = kwargs.get('weight')
        self.const = kwargs.get('cost')  # Minimal cost
        self.connected = []
        self.simple_connected = []
        self.checks = []
        user_prop = kwargs.get('user_properties')
        self.properties=None if user_prop == None else {key:value for (key, value) in user_prop}
        self.walk_probability = None
        self.added_time = kwargs.get('added_time')
        if 'check' not in self.__dict__:
            self.check = 0

    # Index exist
    def isIndex(self, idx):
        return idx in self.index

    #Добавить связь
    def add_connect(self, edge):
        if(isinstance(edge, Edge)):
            self.simple_connected.append(edge.outedge)
            self.connected.append(edge)

    #Add another node to the connected array
    def add_node(self, node):
        self.connected.append(node)

    #На одну больше в счётчике обращений
    def inc(self):
        selr.check += 1

    def node(self):
        self.inc()
        return self.node

    def set_edge(self, edge):
        if(isinstance(edge, Edge)):
            self.connected.append(edge)

    def get_edges(self):
        return self.connected

    def cheapestway(self):
        return 1

    #Пользовательские свойства
    def addProperty(self, prop, value):
        self.properties[prop] = value
    def getProperty(self, prop):
        if prop in self.properties:
            return self.properties[prop]

    '''def addProperties(self, props):
        self.properties={key:value for (key, value) in props}'''


#Пользавательский класс
class Node:
    def __init__(self, node, *args, **kwargs):
        self.node = node
        self.weight = kwargs.get('weight')
        self.structnode = None

    def add_weight(self, weight):
        self.weight = weight

    def add_advanced_node(self, structnode):
        if(isinstance(structnode, StructNode)):
            self.structnode = structnode

    def get_node(self):
        return self.node


class NodePosition(StructNode):
    def __init__(self, x, y, priority, distance):
        StructNode.__init__(None, None, x=x, y=y)
        self.x = priority
        self.y = distance
        self.priority = 0
        self.distance = 0
        self.cost = self.G + self.H  # Оценка пути

    def estimate(self, xOther, yOther):
        goalx = xOther - self.x
        goaly = xOther - self.y
        # Manhattan distance
        d = math.abs(goalx) + math.abs(goaly)

        return (d)


class Edge:
    def __init__(self, inedge, outedge, **kwargs):

        self.outedge = outedge
        self.value = self.outedge
        self.inedge = inedge
        self.weight = kwargs.get('weight')
        self.action = kwargs.get('action')
        self.label = kwargs.get('label')
        self.password = kwargs.get('password')
        #Время добавления дуги
        self.added_time = kwargs.get('added_time')

    def getLabel(self):
        return self.label

    def edge(self):
        return (self.inedge, self.outedge)

    def add(self, outedge):
        self.outedge.append(outedge)

    def change(self, inedge, outedge, **kwargs):
        self.inedge = inedge
        self.weight = kwargs.get('weight')
        self.action = kwargs.get('action')
        self.label = kwargs.get('label')
        self.password = kwargs.get('password')
        self.outedge.append(outedge)

#Helpful class for graph


class HelpGraph:
    def __init__(self, hgrapg):
        self.hgrapg = hgrapg

    def chesk_type(self):
        if not isinstance(self.hgrapg, list):
            return [self.hgrapg]
        return self.hgrapg


#Разные типы графов, где нужны проверки
class OtherGraph:
    def __init__(self, graph):
        self.graph = graph

    def check(self):
        pass


# Save past state of graph
class PastState:
    def __init__(self, node, edges, connectivity=[]):
        self.pastnode = node
        self.pastedge = edges
        self.pastconnection = connectivity


class Graph(AbstractGraph):
    def __init__(self, graphs={}, **kwargs):
        super(Graph, self).__init__(graphs, **kwargs)
        self.graphbase = GraphDict()
        self.paststates = []
        self.last_results = []
        #Зарезервированные свойсива для графа
        self.reserved_properties=['weight', 'node', 'index', 'count']
        self.edgestore=EdgeDict()
        self.edgebase=[]
        if graphs != None:
            for node, edge in graphs.items():
                # self.checknodes(edge)
                self.append_c(node, edge)
        if kwargs.get('Adjacency') != None:
            self.adjacency = Adjacency()

    def Zipped(zipfunc):
        nodes = {}
        for node1, node2 in zipfunc:
            nodes[node1] = [node2]
        return Graph(nodes)
    # Check all nodes for exists

    def checknodes(self, nodes):
        for node in nodes:
            if not self.has_node(node):
                self.add_node(node)

    '''add Graph looks like
       A:[1,2,3]'''
    def append(self, graph):
        if len(graph) == 1:
            key = list(graph.keys())[0]
            if key not in self.graphbase:
                self.graphbase[key] = StructNode(key, self.has_nodes(graph[key]))

    def append_c(self, node, edge, attribute=[]):
        if node not in self.graphbase:
            self.graphbase[node] = StructNode(node, edge, attribute)

    def add_vertix(self, node):
        # main represent of graph
        self.graphbase = {}

    '''Add random connectuon between all nodes E
    count - Number of iters
    arguments: max_weight - maximum random weight

    Add test for the case with similar nodes
    '''
    def add_edge_random(self, count,weight=10, **kwargs):
        from random import choice,randint
        for node in range(count):
            def chice():
                return choice(self.edges())
            self.add_edge(chice().node, chice().node,
                weight =randint(0,weight))

    # Connection between edges (from two sides)
    def connect(self, inedge, outedge, **kwargs):
        self.add_edge(inedge, outedge)
        self.add_edge(outedge, inedge)

    # StructNode is connection
    def add_node(self, node, **kwargs):
        if node in self.graphbase:
            return self.graphbase[node]
        if(not isinstance(node, StructNode)):
            #Добавляем к созданию графа аттрибуты и пользовательские свойства
            def construct_node_user_properties(properties):
                props=[]
                for p in  properties.keys():
                    if p not in self.reserved_properties:
                        props.append((p, properties[p]))
                return props
            #self._add_nodeh(node, construct_node(kwargs))
            self._add_nodeh(node, attribute=kwargs.get('attribute'), index=kwargs.get('index', []),
                user_properties=construct_node_user_properties(kwargs))
        else:
            newnode = node.get_node()
            self.graphbase[newnode] = StructNode(newnode, added_time=time.ctime())

        return self.graphbase[node]

    def _add_nodeh(self, node, **kwargs):
        if node in self.graphbase:
            count = self.graphbase[node].check
            self.graphbase[node] = StructNode(node, [],
                                              attribute=kwargs.get('attribute'), index=kwargs.get('index', []),
                                              checks=count + 1,
                                              user_properties=kwargs.get('user_properties'),
                                              added_time = time.ctime())
        else:
            #print('NA :', node)
            self.graphbase[node] = StructNode(node, [],
                                              attribute=kwargs.get('attribute'), index=kwargs.get('index', []),
                                              checks=0, user_properties=kwargs.get('user_properties'),
                                              added_time=time.ctime())

    '''def add_node_from(self, imps):
        yield from imps'''

    # Add node and in the the absence case, raise Exception
    def add_node_exc(self, node, exception="StructNode already in graph"):
        if node in self.graphbase:
            raise GraphError(exception)
        self.add_node(node)


    '''agruments:
    ffilter - func filter for generate nodes
    start - start position in graph
    increment - generate step
    end - end of generate nodes. defaule value - 1000'''
    def add_nodes(self, nodes,**kwargs):
        start = kwargs.get('start',0)
        end = kwargs.get('end', 1000)
        step = kwargs.get('step',1)
        func_filter = kwargs.get('ffilter')

        #Simple version of (start, step, end)
        ranges = kwargs.get('f')
        if ranges != None and len(ranges)==3:
            for i in range(ranges[0], ranges[1], ranges[2]):
                self.add_node(i)
        if len(nodes) > 0:
            for node in nodes:
                self.add_node(node)
        else:
            for i in range(start, end, step):
                self.add_node(i)

    def delete_node(self, node):
        if node in self.graphbase:
            self.paststates.append(PastState(node, self.graphbase))
            del self.graphbase[node]


    '''Edge area'''

    def delete_edge(self, nodein, nodeout):
        if nodein in self.graphbase and nodeout in self.graphbase:
            self.graphbase[nodein].value.remove(nodeout)


    def nodes(self):
        return list(self.graphbase.values())

    def rename_node(self, node, newname):
        if self.has_node(node):
            temp = self.graphbase[node]
            del self.graphbase[node]
            self.graphbase[newname] = temp
            for ch in self.nodes():
                if ch.node == node:
                    ch.node = newname
                else:
                    if len(ch.connected) > 0:
                        #dove = list(filter(lambda x: x.inedge.node == node, ch.connected))
                        for conn in ch.connected:
                            if conn.outedge.node == node:
                                conn.outedge.node = newname  

    #Соседние вершины у ноды соединённые отрезком
    #!Дописать
    def neighbors(self, node):
        return self.graphbase[node]

    def edge(self, node1, node2,*args, **kwargs):
        '''return current edge from edgebase'''
        #print(node, node in self.graphbase)
        assert(node1 in self.graphbase and node2 in self.graphbase)
        self.edgebase.append(Edge(node1, node2, args))


    def has_edge(self, node1, node2):
        '''
            Check if between node1 and node2 has edge
        '''
        if isinstance(node1, StructNode) and isinstance(node2, StructNode):
            return self._has_edge_inner(node1.node, node2.node)
        if isinstance(node1, StructNode):
            return self._has_edge_inner(node1.node, node2)
        if isinstance(node2, StructNode):
            return self._has_edge_inner(node1, node2.node)
        return self._has_edge_inner(node1, node2)

    def _has_edge_inner(self, node1, node2):
        if self.has_node(node1) and self.has_node(node2):
            return node2 in \
            list(map(lambda x: x.outedge.node, self.graphbase[node1].connected))
        return False

    '''arguments:
    rev - pair connect between nodes'''
    def add_edge(self, inedge, outedge, **kwargs):
        reverse = kwargs.get('rev')

        #In case when edge not exists, create it
        node1 = self.check_and_create(inedge)
        node2 = self.check_and_create(outedge)
        if not self.has_edge(node1, node2):
            # assert self.has_nodes(HelpGraph(outedge).chesk_type())]
            self.graphbase[inedge].add_connect(Edge(node1, node2, **kwargs))
            if reverse != None:
                self.graphbase[outedge].add_connect(Edge(node2, node1,**kwargs))


    '''Input params - StructNodes'''
    def add_edge_from_nodes(self, inedge, outedge,**kwargs):
        self.add_edge(inedge.node, outedge.node)


    def show_graph(self):
        for graph, node in self.graphbase.items():
            print(graph, node.value)

    ''' Check area '''


    # high ordered func check
    def has_node_bind(self, check, node):
        return check(node)

    def has_node(self, node):
        return node in self.graphbase

    def has_nodes(self, nodes):
        return [node for node in nodes if self.has_node(node)]

    def get_graph(self):
        return self.graphbase

    #Выбрать случайный нод
    def get_random_node(self):
        from random import choice
        return choice(list(self.graphbase.keys()))

    #return node
    def __getitem__(self, node):
        if self.has_node(node):
            return self.graphbase[node]

    #Взвешанный ли граф
    def is_weight(self):
        return len([node for node in self.graphbase if self.graphbase[node].weight == None]) == 0

    #Сортировка связей
    def sort_edges_by_weight(self):
        return sorted(self.edges(), key=lambda x: x.weight)

    # Oprional. Восстановить если возможно
    #Нужен тест
    def recovery(self, key):
        result = list(filter(lambda x: x.pastnode, self.paststates))
        if len(result) > 0:
            self.graphbase[result[0].pastnode] = StructNode(result[0].pastnode, [])

    #Поиск по индексу
    def findIndex(self, idx):
        return [node for node in self.graphbase
                if idx in self.graphbase[node].index]

    # Size of Graph
    def size(self):
        return len(self.graphbase)

    #Проверить, существует ли нода и если нет, то создать
    def check_and_create(self, node):
        return self.add_node(node)

    #Возвращает уникальные ноды
    def unique_nodes(self):
        return (set(self.get_graph()))

    def adjacent(self, node):
        if not self.has_node(node):
            raise Exception("StructNode {0} if not found"
                            .format(node))
        return self.graphbase[node].value

    def nodeinfo(self, node):
        if node in self.graphbase:
            return self.graphbase[node]

    def set_weight(self, edge_in, ed, wid):
        self.graphbase[edge_in].set_edge(Edge(edge_in, ed, weight=wid))

    #Get weight os current node


    def _checkGraphType(self, somegraph):
        if(not isinstance(somegraph, Graph)):
            raise GraphException("This is not type of Graph")

    #Implementation of cartesian product
    def __mul__(self, another_graph):
        self._checkGraphType(another_graph)
        product = GraphProduct()
        return product.cartesian(self, another_graph)

    def tensorProduct(self, another_graph):
        self._checkGraphType(another_graph)
        tensor = GraphProduct()
        return tensor.tensor(self, another_graph)


    # Проверка на циклы
    # http://code.google.com/p/python-graph/source/browse/trunk/core/pygraph/algorithms/cycles.py
    # nodes - {'A':['B','C']}
    # http://en.wikipedia.org/wiki/Tarjan%E2%80%99s_strongly_connected_components_algorithm
    def has_cyclic(self, nodes, another):
        spanning_tree = []
        if not has_nodes(nodes):
            while nodes != another:
                if nodes is None:
                    return []
                spanning_tree.append(nodes)
        return spanning_tree


    #Является ли текущий граф деревом
    #1. Ациклический граф
    #2. Любые две вершины связыны простой цепью
    #3. Число рёбер меньше на 1 числа вершин
    def is_tree(self):
        onenode = self.ndoes() - self.edges()
        if onenode != 1:
            return False
        for n in self.nodes:
            if(len(n.connected) != 1):
                return False

        return True

    def isBipartite(self):
        q = Queue()
        key = self.graphbase.keys()
        start = list(key)[random.randint(0, len(self.graphbase) - 1)]
        for node in self.graphbase[start]:
            print(node)

    #Добавить веса графа
    def add_weight(self, node, weight):
        if node in self.graphbase:
            self.graphbase[node].add_weight(weight)

    def filter_node(self, pattern):
        return list(filter(lambda x: pattern(x), self.graphbase))

    def __str__(self):
        return "This graph has {0} ver and {1} nodes"\
            .format(len(self.graphbase), len(self.nodes()))

    def connected(self, node):
        return self.graphbase[node].connected

    def connected_counts(self):
        for node in self.graphbase.items():
            yield node

    def clear(self):
        self.graphbase.clear()

    # Query for graph attributes
    def query(self, **kwargs):
        kwargs.get('select')

    """load area from file
    *Plain load from text file
    *Load from JSON"""

    #Загрузка из файла обычного формата
    #Формат вида
    #1 2 3
    #2 1 3
    #3 1
    def load(self, filename):
        currentmap = {}
        data = self._IO(open, IOError, 
            'Failed to open this file', filename).readlines()
        adding = lambda data: self.add_node(data[0])
        result = [adding(i.split()) for i in data]
        return open(filename).read()

    #Загрузка графа из файла формата json
    def loadfromJSON(self, jsonfilename):
        data = self._IO(json.loads, ValueError, 
            'Failed to open this jsonfile', jsonfilename)
        json.loads(jsonfilename)
        "Test here"

    #Что-то вроде монады
    def _IO(self, func, expname, message, *args):
        try:
            return func(*args)
        except expname:
            raise expname(message)

    '''Compute all values for choice node
    node - this node. Check or create
    func - filter function for compute nodes

    example:
    edge('a','b',10)
    edge('a','c',8)
    edge('b','d',12)
    edge('c','d',5)
    set_simple_rank('e', weight > 8) => e = 22(a,b)
    '''
    def set_simple_rank(self, node, func):
        #print(self.edge())
        return list(filter(lambda x:func(x), self.nodes()))


    def schematic(self, expr):
        sc = SchematicGraph()
        sc.add(expr)

    def get_edges(self):
        '''
            Return all pair of edges in human view
        '''
        for node in self.graphbase.values():
           yield(node.node, list(map(lambda x: x.outedge.node, node.connected)))

class NewNode:
    def __init__(self, node1, node2, newnode):
        self.node1 = node1
        self.node2 = node2
        self.newnode = newnode

#http://en.wikipedia.org/wiki/Graph_product
class GraphProduct():
    def __init__(self):
        self.minv = 100
        self.maxv = 999

    def _createNewNodes(self, graph1, graph2):
        '''
            Create newnodes as product of two node names in graph
            For example: Node A and Node B will be Node AB
        '''
        newnodes = []
        #Check types and count of nodes
        nodes1 = graph1.nodes()
        nodes2 = graph2.nodes()
        if len(nodes1) == 0 or len(nodes2) == 0:
            raise GraphException("Product graph: Count of nodes is zero in one of graph")
        if not isinstance(nodes1[0], StructNode) or not isinstance(nodes2[0], StructNode):
            raise GraphException("Product graph: type of nodes is not StructNode")
        for node in graph1.nodes():
            for node2 in graph2.nodes():
                if isinstance(node.node, str) and isinstance(node2.node, str):
                    newnodes.append(NewNode(node, node2, node.node+node2.node))
        return newnodes

    def _preCartesian(self, graph1, graph2):
        '''
            Rename duplicate nodes
        '''
        for node in graph2.nodes():
            if graph1.has_node(node.node):
                graph2 = self._renameNode(graph2, node.node, node.node)
        return graph2

    def cartesian(self, graph1, graph2):
        graph2 = self._preCartesian(graph1, graph2)
        newgraph = Graph()
        newnodes = self._createNewNodes(graph1, graph2)
        [newgraph.add_node(newnode.newnode) for newnode in newnodes]   
        for oldnode in newnodes:
            for conn1 in oldnode.node1.connected:
                newgraph.add_edge(oldnode.newnode,  conn1.outedge.node + oldnode.node2.node)
                newgraph.add_edge(conn1.outedge.node + oldnode.node2.node, oldnode.newnode)
            for conn1 in oldnode.node2.connected:
                newgraph.add_edge(oldnode.newnode,  oldnode.node1.node+ conn1.outedge.node)
                newgraph.add_edge(oldnode.node1.node+ conn1.outedge.node, oldnode.newnode)
        return newgraph

    def tensor(self, graph1, graph2):
        '''
            Result is bipartite double cover

        '''
        graph2 = self._preCartesian(graph1, graph2)
        newgraph = Graph()
        newnodes = self._createNewNodes(graph1, graph2)
        for oldnode in newnodes:
            conn1 = list(map(lambda x: x.outedge.node, oldnode.node1.connected))
            conn2 = list(map(lambda x: x.outedge.node, oldnode.node2.connected))
            for c1 in conn1:
                for c2 in conn2:
                    newgraph.add_edge(oldnode.newnode, c1 + c2, rev=True)
        return newgraph

    def _cart_connect_new_graph(graph1, graph2):
        pass

    def _renameNode(self, graph, oldnode, node):
        '''
            In the case if graph already contain node with the same name
        '''
        if graph.has_node(node):
            return self._renameNode(graph, oldnode, node + str(random.randint(self.minv\
                ,self.maxv)))
        graph.rename_node(oldnode, node)
        return graph


#Add with schematic
#a->b a connected to b
#a<->b  a connected to b and b connected to a
#a-10->b a connected to b with weight is 10

class SchematicGraph:
    def __init__(self):
        pass

    def add(self, expr):
        """
        Connect a to nodes b and c
        add('a->[b,c]')
        """
        newgraph = Graph()
        rush = []
        arrs=[] #Для массива
        let_arr = False
        expr = expr.replace(' ','')
        expectedNode = None
        for i in range(len(expr)):
            #Если символ, то создаём новую ноду
            if expr[i] == '-' and expr[i+1] == '>':
                if '_' not in rush:
                    newgraph.add_node(expr[i-1])
                    newgraph.add_node(expr[i+1])
                    if expectedNode != None:
                        newgraph.add_node(expectedNode)
                        expectedNode = None
                else:
                    #Берём всё ноды и соединём с установленной
                    for nodes in newgraph.nodes():
                        newgraph.add_edge(nodes.node, expr[i+1])
            elif expr[i] == '-' and re.search('[0-9]', expr[i+1]) != None:
                print(re.search('[0-9]', expr[i+1]).group(0))
            elif expr[i] == '[':
                endbrk = expr.find(']',i)
                for name in re.findall('[a-z]', expr[i+1:endbrk]):
                    newgraph.add_node(name)

            elif(expr[i] == '_'):
                rush.append('_')

            elif expr[i] == '<' and expr[i+1] == '-':
                expectedNode = expr[i-1]
        return newgraph

    def run(self, command):
        """ run command """
        """ 
        *d - get degree of d
        """
        pass


# Hypergraph area
#http://darwin.bth.rwth-aachen.de/opus3/volltexte/2011/3645/pdf/3645.pdf
class HyperNode:
    """
    Node for hypergraph
    """
    def __init__(self, node_name, group_name):
        self.node_name = node_name
        #all groups contains this node
        self.groups=[group_name]

    def addGroup(self, name):
        self.groups.append(name)

class HyperGraph(Graph):
    def __init__(self):
        super(Graph, self).__init__()
        self.node_neighboards = {}
        #Записать это в виде структуры
        self.graph = []
        self.nameset = []
        self.edgeset = []
        self.nodeset = []
        self.groups = {}
        self.nodedict = {}

    def neighboards(self, node):
        self.node_neighboards[node] = {}

    def addNode(self, groupname, node):

        if groupname not in self.groups:
            raise Exception("This groupname not in groups")
        self.groups[groupname].append(node)
        self._addDict(node, groupname)

    #Проверить сущестование нода в словаре
    def _addDict(self, node, groupname):
        if node not in self.nodedict:
            self.nodedict[node] = HyperNode(node, groupname)
        else:
            self.nodedict[node].addGroup(groupname)

    #Добавить нод без группы
    def addAloneNode(self, node):
        self.nodeset.append(node)

    '''nameset A - space name
    nodeset B,C - node who contains in A
    http://sharpen.engr.colostate.edu/mediawiki/index.php/Hypergraph_implementation
    http://stackoverflow.com/questions/8348459/is-there-a-library-that-provides-a-directed-hypergraph-implementation-in-c
    '''
    def add_node_set(self, namespace, nodeset):
        if namespace not in self.nameset:
            self.nameset.append(namespace)
        self.nameset[namespace] = nodeset

    def add_edge(self, nodename, edge, nodespace='default'):
        if(nodespace == 'default' and len(self.nameset)) > 0:
            nodespace = self.nameset[0]

        self.edgeset[nodename] = edge

    def addGroup(self, groupname):
        self.groups[groupname] = []
    
    def delNode(self, groupname, node):
        if groupname not in self.groups:
            raise Exception("This groupname not in groups")
        self.groups[groupname].remove(node)

    #Balanced hypergraph contains contains no strong cycle of odd length.
    #Также, одинаковое количество элементов в группе
    def is_balanced(self):
        return len(self.find_cycle()) % 2 != 0

    #Найти оптимальные алгоритмы для нахождения циклов
    def find_cycle(self):
        if len(self.groups) == 0:
            raise Exception("Length of groups is zero")
        notvisited = list(self.groups.keys())
        startgroup = list(self.groups.keys())[0]
        nodes = self._getNodesFromGroup(startgroup)
        #Считаем количество групп
        def inner(startgroup):
            countGroups = 0
            path = []
            while len(notvisited) > 0:
                path.append(startgroup)
                nodes = self._getNodesFromGroup(startgroup)
                notvisited.remove(startgroup)
                if len(nodes) > 0:
                    current_groups = self.nodedict[nodes[0]].groups
                    startgroup = None
                    for group in current_groups:
                        if group in notvisited:
                            startgroup = group
                    if startgroup == None:
                        return path
            return path
        return inner(startgroup)
        #nodes = self._getNodesFromGroup(startgroup)
        #print(self.nodedict[nodes[0]].groups)

    #Get nodes from groups(random)
    def _getGroups(self, group):
        return self.nodedict[random.choice(self.groups[group])].groups

    def _getNodesFromGroup(self, group):
        return self.groups[group]

    #Ищем связь между двумя группами
    #Make private
    def findConnectionBetweenGroups(self, group1, group2):
        return list(set(self.groups[group1]) & set(self.groups[group2]))


class RandomGraph(Graph):
    def __init__(self, num, prob):
        self.g = self._constructGraph(num, prob)

    def constructAdjMatrix(self, num, prob):
        '''
        Example for construct graph
        '''
        return np.random.binomial(1, self._prob, (self._num, self._num))

    def _constructGraph(self, num, prob):
        self.g = Graph()
        self.g.add_nodes(range(1, num+1))
        for n in range(1,num+1):
            for n2 in range(1, num+1):
                if n != n2 and np.random.binomial(1, prob) != 0:
                    self.g.add_edge(n, n2)
        return self.g

    def mostConnectedNode(self):
        '''
            return most connected node from random graph
        '''
        edges = self.g.get_edges()
        return sorted(lambda x:len(x.connected), edges)

    def getEdges(self):
        return self.g.get_edges()

#Граф интересов (посмотреть)
class InterestGraph(Graph):
    #nums - count of people
    def __init__(self, nums):
        super(Graph, self).__init__()
        self.nums = nums

    #Добавить интерес для конкретного человека
    def addInterest(self, pos, interest):
        self.add_node(pos, interest)

#Exception for the graph
class GraphException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


#h = HyperGraph()
#h.add_node_set("A", ["B","C"])

# b->q
# b<->q
# b/q   и не включает q
# _->a  Каждый элемент графа, соединён с a 
def test_schematic():
    schem = SchematicGraph()
    #gr = schem.add('a->[b,c,d,e]')
    gr2 = schem.add('a-10->b')
    #Соединить все вершины с q
    #gra = schem.add('_->q')
    #Соединить q со всеми вершинами _->q
    if(isinstance(gra, Graph)):
        print("This is new graph")

def test_schematic2():
    schem = SchematicGraph()
    gr = schem.add('c <- a -> [b,c,d,e]')
    for n in gr.get_graph():
        print(n, )

def test_hypergraph():
    h = HyperGraph()
    h.addGroup('Red')
    h.addGroup('Green')
    h.addGroup('Blue')
    h.addGroup('Yello')
    h.addGroup('Black')
    h.addNode('Red', 'x1')
    h.addNode('Red', 'x2')
    h.addNode('Red', 'x3')
    h.addNode('Green', 'x3')
    h.addNode('Black', 'x1')
    h.addNode('Green', 'x4')
    h.addNode('Blue', 'x4')
    h.addNode('Blue', 'x5')
    h.addNode('Yello', 'x5')
    h.addNode('Yello', 'x6')
    h.addNode('Black', 'x6')
    #h.findConnectionBetweenGroups('Red', 'Green')
    print(h.is_balanced())