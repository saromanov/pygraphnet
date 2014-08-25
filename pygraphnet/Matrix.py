import numpy as np

#Some implementations of matrix with graphs

class LaplacianMatrix:
	def __init__(self):
		self.matrix = np.matrix([])
	def addEdge(self, node1, node2):
		pass


#Матрица популярности, чем значение увеличивается, кода ищется(или другое) нод

class PopularMatrix:
	def __init__(self):
		self.matrix = np.matrix(np.zeros(10) * 5)

	def incNode(node, pos):
		position = self.matrix[node][pos]
		if position == 0: position = 1
		else: position += 1


#Construct adjacency matrix
#Матрица смежности
class Adjacency:
    def __init__(self):
        self.data = np.zeros((1,1))
        self._nodes = []
    
    def add_node(self, nodename):
    	self._nodes.append(nodename)
    	if len(self._nodes) > 1:
    		self.data = np.vstack([self.data, np.zeros((1, self.data.shape[0]))])
    		self.data = np.append(self.data, [[0] for i in range(self.data.shape[0])], 1)

    def add_edge(self, node1, node2):
    	self.data[self._nodes.index(node1)][self._nodes.index(node2)] = 1
