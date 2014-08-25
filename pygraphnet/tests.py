import graph
import algorithms
import numpy as np
import unittest

#Добавить профилер
#Добавить анализ кода

#Тестирование стандартного графа
class GraphTest(unittest.TestCase):
	pass


#a -> [0.8,0.7] -> b

def create_graph_1():
	g = graph.Graph()
	g.add_node('A')
	g.add_node('B')
	g.add_node('C')
	g.add_node('E')
	g.add_edge('A', 'B', weight=10)
	g.add_edge('A', 'C', weight=5)
	g.add_edge('B', 'E', weight=4)
	g.add_edge('E', 'A', weight=8)
	g.add_edge('C', 'B', weight=15)
	#print(g.edges())

	n = graph.Edge('a', 'b', weight=10)
	print(g['A'].connected)
	return g

def create_hypergraph():
	h = graph.HyperGraph()
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
	return h

class HyperGraph(unittest.TestCase):
	def test_create(self):
		h = create_hypergraph()
	def test_balanced(self):
		h = create_hypergraph()
		self.assertTrue(h.is_balanced())


#Создание графа с помощью схемы
class SchematicGraph(unittest.TestCase):
	# b->q
	# b<->q
	# b/q   и не включает q
	# _->a  Каждый элемент графа, соединён с a 
	def test_schematic():
		schem = SchematicGraph()
		gr = schem.add('a->[b,c,d,e]')
		#Соединить все вершины с q
		gra = schem.add('_->q')
		#Соединить q со всеми вершинами _->q
		if(isinstance(gra, Graph)):
			print("THis is new graph")
			print(gr.nodes())

class AlgorithmsTest(unittest.TestCase):
	def test_belmann_ford(self):
		sgraph = create_graph_1()
		algo = algorithms.Algorithms(sgraph)


if __name__ == '__main__':
	#unittest.main()
	sgraph = create_graph_1()