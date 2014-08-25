import graph as g
import numpy

#Dynamic algorithms
#Dynamic connectivity, Shortest path
#http://www.disp.uniroma2.it/users/italiano/dynamic1.pdf

#Обновление операций быстро как только возможно
class Dynamic:
	def __init__(self, sgraph):
		pass
	def update(self, value):
		pass
	#return the distance between x and y
	def query(self, x, y):
		pass
	def start(self, *args, **kwargs):
		pass


#Возвращает значение, существует ли отрицательный цикл
#Если существует, то возвращаем false, а если нет, то
#кратчайший путь
class BellmannFord(Dynamic):
	def __init__(self, graph):
		self.graph = graph


	def start(self):
		start_node = self.graph.get_random_node()
		matrix = numpy.matrix((1,2))

		print(matrix)
		for node in self.graph.nodes():
			for current_node in node.connected:
				pass
		return True


#http://urban-sanjoo.narod.ru/bellman-ford.html
def test_belmann_ford():
	gr = g.Graph()
	gr.add_nodes(['s','t','x','z','y'])
	gr.add_edge('s', 't', weight=6)
	gr.add_edge('s', 'y', weight=7)
	gr.add_edge('t', 'x', weight=5)
	gr.add_edge('t', 'z', weight=-4)
	gr.add_edge('t', 'y', weight=8)
	gr.add_edge('z', 'x', weight=7)
	gr.add_edge('y', 'x', weight=2)
	gr.add_edge('x', 't', weight=-2)
	BellmannFord(gr).start()


test_belmann_ford()