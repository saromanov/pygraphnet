import numpy
import graph

#http://www.cs.huji.ac.il/~nir/Papers/FLNP1Full.pdf

#Различные алгоритмы из машинного обучения

#https://github.com/opcode81/ProbCog/blob/master/src/probcog/bayesnets/core/BNDatabase.java
class BayesianDB:
	def __init__(self, path):
		self._path = path
		self._results = {}

	def open(self):
		f = open(self._path)
		with f:
			lines = f.readlines()
			for line in lines:
				self._results['test'] = line

class BayesianStruct:
	def __init__(self,name, cpd, **kwargs):
		#CPD between 0 anf 1
		self.cpd = cpd
		self.name = name
		self.connecned=[]

	def calc_cpd(self, bstrcuts):
		if isinstance(bstrcuts, BayesianNetwork):
			for node in bstrcuts.connecned:
				self.cpd *= node.cpd

class BayesianNetworkError(Exception):
	def __init__(self, message):
		self.message = message

'''Nums  - count of edges
states looks like map {name:probability}'''
class BayesianNetwork:
    def __init__(self):
        self.bgraph = graph.Graph()
        #Проверка, создана ли сеть
        self.isConstructed = False
        self.connectedNodes=[]

    def addNode(self, name, *args, **kwargs):
    	self.bgraph.add_node(name, cpd=kwargs.get('cpd', 0.0))

    #list_of_nodes -> create undirected graph
    def addNodes(self, list_of_nodes):
    	pass

    def connect(self, node1, node2):
    	self.connectedNodes.append(node2)
    	self.bgraph.add_edge(node1, node2)

    def __mul__(self, bayesianlist):
    	pass

    #Создание Байесовский сети и запись результатов в map
    def _constructNetwork(self):
    	if not self.isConstructed:
    		#Ищем входные ноды (ноды, к которым не идут связи)
    		network = ConstructBayesianNetwork(self.bgraph, self.connectedNodes)
    		nodes = network._getStartNodes()
    		return nodes


    ##Show Statements
    #Independence http://www.cs.columbia.edu/~kathy/cs4701/documents/conditional-independence-bn.txt
    def show_conditional_independence(self):
    	result = self._constructNetwork()
    	return 'I({0})'.format(';'.join(map(lambda x:x.node, result)))

    def learn(self):
    	return LearningBayesianNetwork(self.bgraph)


class ConstructBayesianNetwork:
	def __init__(self, bgraph, connectedNodes, *args, **kwargs):
		self.bgraph = bgraph
		self.connectedNodes = connectedNodes
	def _getStartNodes(self): return list(filter(lambda x:x.node not in self.connectedNodes, self.bgraph.nodes()))

#All of classes 
class LearningBayesianNetwork:
	def __init__(self, bgraph):
		self.bgraph = bgraph

	#Calculate marginal probability
	def marginals(self):
		return sum(map(lambda x:x.getProperty('cpd'), self.bgraph.nodes()))

class MarkovStruct:
	def __init__(self):
		self.name = 'Default'
		self.count = 0
		self.states=[]
		self.connected=map()


class ConstructMarkovGraph:
	def __init__(self):
		self._nodes={}
		self._edges={}
class MarkovNetwork:
	def __init__(self, sigma, states):
		self._states = states
		self._sigma = sigma

	def createFactor(self, states, params):
		Factor.FactorManager(states, params)

	def isIndependence(self, factor1, factor2):
		pass


def test_markov_network():
	m = MarkovNetwork()
	m.createFactor([2,2], [0.1,0.2,0.1])


def test_create_bn():
	bn = BayesianNetwork()
	bn.addNode('E',cpd=0.5)
	bn.addNode('A', cpd=0.3)
	bn.addNode('B', cpd=0.7)
	bn.addNode('D', cpp=0.3)
	bn.addNode('C', cpd=0.4)
	bn.connect('E', 'B')
	bn.connect('A', 'B')
	bn.connect('A', 'D')
	bn.connect('B', 'C')
	return bn

#http://www.cs.iastate.edu/~jtian/cs673/cs673_spring05/lectures/cs262a-4.pdf Стр13
def test_bayesian():
	bn = test_create_bn()
	print(bn.show_conditional_independence())
	#Изменения, которые вносятся в сеть, не будут вноситься в обучение
	learn = bn.learn()

	#print(bn.show_conditional_independence())
	#bn.addNode('Flu', 0.0)-(('tt', .5), ())

def test_marginals():
	bn = test_create_bn()
	learn = bn.learn()
	return learn.marginals()

print(test_marginals())
#test_bayesian()