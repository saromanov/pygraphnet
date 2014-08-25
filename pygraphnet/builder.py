import graph

#Создание своих графов


class BuildGraph:
    def __init__(self,nodes, vertices, *args,**kwargs):
        self.graph = {}
        self.graph_type = kwargs.get('gtype')

    def add_algorithm(self, *args,**kwargs):
        pass

    def add_type(self, *args,**kwargs):
        self.graph_type = kwargs.get('gtype')
        if(self.graph_type =='dual'):
            raise NotImplemented

    #Show Current Graph
    def output(self):
        return self.graph

    def _make_dual_graph(self):
        pass

    def create(self):
        gr = graph.Graph()
        for node in range(nodes):
            gr.add_node(node)

class MakeGraph:
    def __init__(self, nodes):
        self.count_nodes = nodes

    def _addNodes(self, item):
        self.nodes.append(item)

    def create(self):
        pass

class StructTree:
    def __init__(self, rightNode, leftNode):
        self.rightNode = rightNode
        self.leftNode = leftNode

#Base class for create tree
class MakeTree:
    '''Optional parametres
    levels - count of levels in tree
    if will be add in tree more then levels, will be error
    '''
    def __init__(self, root, *args, **kwargs):
        self.root = root
        self.base = {}
        self.levels = kwargs.get('levels', 0)
        self.structureBase ={}

     #Pair nodes
     #f.ex ['a':'b','c']

    def addNodes(self, root, rightNode, leftNode):
        if (not self._check(root)  or len(self.structureBase) == 0 )and not self._check(rightNode) and not self._check(leftNode):
            self.structureBase[root] = StructTree(rightNode, leftNode)
            self.base[root] = [rightNode, leftNode]
        else:
            raise NameError()

    def _check(self, item):
        return item in self.base

    def checkNode(self, item):
        return self._check(item)

class Shape:
    def __init__(self, graph):
        self.graph = graph

    def rectangle(self, *args, **kwargs):
        israndom = kwargs.get('rand')
        if len(self.graph) >= 4:
            rect = self.randsample(4)
            temp = rect
            graph = {i: None for i in rect}
            while len(rect) > 0:
                first = rect.pop()
                rect.reverse()
                last =rect.pop()
                rect.reverse()
                graph[first] = [last]
                graph[last] = [first]
            keys = list(graph.keys())
            graph[keys[0]].append(keys[1])
            graph[keys[1]].append(keys[0])
            graph[keys[2]].append(keys[3])
            graph[keys[3]].append(keys[2])
        return graph


    def trangle(self):
        pass

    def randsample(self, nums):
        return random.sample(self.graph,nums)

class BullGraph(MakeGraph, Shape):
    MAX=5
    def __init__(self):
        super(MakeGraph, self).__init__()
        self.graph={i: i for i in range(self.MAX)}
        self.numbers = self.rand()
        self.endnumbers = self.rand()
        self.make_edges(2, [self.numbers[0], self.numbers[1]])
        self.make_edges(self.numbers[0], [2, self.numbers[1]])
        self.make_edges(self.numbers[1], [2, self.numbers[0]])
        self.make_edges(self.rand()[0], self.rand()[1])
        self.make_edges(self.rand()[0], self.rand()[1])
        super(Shape, self).__init__(self.graph)
        print(self.graph)

    def rand(self):
        return random.sample([0,1,3,4],2)

    def make_edges(self, main, output):
        self.graph[main] = output


class Tree(list):
    def __init__(self, num,**kwargs):
        self.num = num
        self.attributes=kwargs
        self.name=kwargs.get('name', 'tree')



#Multi-dimension tree
#Kd trees for cheap learning
class KDNode:
    def __init__(self, name, x, y, *args, **kwargs):
        self.x = x
        self.y = y
        self.name = name

#Разделяющая линия
class KDLine:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos


#Возможно, points - это координата (x,y)
class KDTree:
    def __init__(self, points, dims):
        self.points = points
        self.dims = dims
        self.sgraph = graph.Graph()
        self._build(self.points, d0)
        self.lines = []

    def _build(self, point, dim):
        if len(self.points) == 1:
            print("Only one point")
            return 
        if self.dims % 2 == 0:
            #Split to vertical line
            self.sgraph.add_node(point)
            self.sgraph.add_node(self._create_line(True))
        else:
            #Split to horizontal line
            self.sgraph.add_node(point)
            self.sgraph.add_node(self._create_line(False))

    #Создаём линию разделения
    #pos - True (vertival)
    #False - Horizontal
    def _create_line(self, pos):
        if len(self.lines) == 0:
            self.lines.append(KDLine("l1", pos))
            return "l1"
        else:
            line_name = "l%d".format(len(self.lines))
            self.lines.append(KDLine(line_name, pos))
            return line_name


    def addItem(self, item):
        for i in self.dims:
            pass

    def distance(self, node1, node2):
        ''' Расстояние между элементами в дереве'''
        return sqrt(sum(map(lambda x: (node1[x] - node2[x])**2, sequence)))

'''n-dimensional graph'''
class DeBruijn(object):
    def __init__(self, dim):
        self.dim = dim

    def make(self):
        pass
        '''states=[0,1,00,11,000,111,010,011,001]
        for node in range(self.count):
            pass'''


        

#Graph only with cycles
#Floyd cycle finding algorithm
class CycleGraph(MakeGraph):
    def __init__(self, nodes, num):
        self.nodes = nodes #Count of nodes
        self.num = num #Количество вершин в цикле
        assert(nodes >= num and num > 1)
        self.bgraph = graph.Graph()

    def create(self):
        super(CycleGraph, self).create()
        for node in range(self.nodes):
            self.bgraph.add_node(node)

    def _create_cycle(self):
        while not self._check_cycle():
            pass

    def _check_cycle(self):
        return True


#Планарный граф
#Может быть изображён на плоскости без пересечения рёбер

'''example:
add_node(a)
add_node(b)
add_node(c)
add_node(d)
add_edge(a,b)
add_edge(a,c)
add_edge(a,d)
add_edge(d,c)
add_edge(d,b)
add_edge(b,c)
'''

##Course 6.889: Algorithms for Planner Graphs and Beoynd (Fall 2011)
class PlanarGraph:
    def __init__(self):
        self.sgraph = graph.Graph()
        self.planeSet={}

    #plane - Находящийся уровень
    def add_node(self, node, plane):
        self.sgraph.add_node(node)
        self.create_plane(plane, node)

    def add_edge(self, node1, node2):
        self.sgraph.add_edge(node1, node2)

    #Высчитываем ранг ноды, если он меньше на единицу количества нодов
    #То граф не планерный???
    def get_rank_edge(self):
        pass

    #Вводим дополнительную плоскость
    def create_plane(self, plane, node):
        #Три ноды на одном уровне и одна на другом
        if plane not in self.planeSet:
            self.planeSet[plane] = [node]
        else:
            res = self.planeSet[plane]
            res.append(node)
            self.planeSet[plane] = res

    def check_plane(self):
        return len(self.planeSet) > 1

    #Минор графа
    def checkMinor(self):
        #edge contraction
        pass

#http://web.engr.oregonstate.edu/~erwig/papers/PersistentGraphs_IFL97.pdf
class PersistentGraph:
    def __init__(self, data):
        #Functional array for storing adjacency lists and node labels
        self.funcarray = {}
        #node with positive integer - conatins in graph
        #node with negative integer - deleting from graph
        self.store={}

    def add_node(self, nodename):
        self.store[(nodename,1)] = []

    def delete_node(self, nodename):
        if (nodename,1) in self.store:
            data = self.store[(nodename, 1)]
            del self.store[(nodename,1)]
            self.store[(nodename,0)] = data
            print(self.store)

class BinaryTree(MakeTree):
    def __init__(self, root, *args, **kwagrs):
        super(BinaryTree, self).__init__(root, args, kwagrs)
    def create(self):
        return self.root



def test_planner_graph():
    pc = PlanarGraph()
    pc.add_node('a',1)
    pc.add_node('b',1)
    pc.add_node('c',2)
    pc.add_node('d',1)
    pc.add_edge('a','b')
    pc.add_edge('a','c')
    pc.add_edge('a','d')
    pc.add_edge('d','c')
    pc.add_edge('d','b')
    pc.add_edge('b','c')
    print(pc.check_plane())

test_planner_graph()
#b = BuildGraph(2,5).create()
#cyc = CycleGraph(5,2).create()

p = PersistentGraph(1)
p.add_node(10)
p.delete_node(10)
