import graph
import threading
from random import randint
from multiprocessing import Pool, Process

'''Experimental, ideas, etc'''


#Через какое время будет изменяться веса графа

def change_weight(grap):
	grap.set_weight('A','B', randint(1,100))

#Веcа графа, изменяются со временем
class TimeGraph:
	def __init__(self, nums):
		self.nums = nums
		self.gr = graph.Graph()

	def compute_graph(self):
		self.gr.add_node('A')
		self.gr.add_node('B')
		self.gr.add_node('C')
		self.gr.set_weight('A','B',5)
		self.gr.set_weight('A','C',7)
		self.gr.set_weight('B','C',2)

	def start(self):
		p = Process(target=change_weight, args=(self.gr,))
		p.start()
		p.join()

	def get_weight(self):
		self.gr.get_weight('A','B')


class PNode:
	def __init__(self, node, conn):
		self.node = node
		self.conn = conn

class StateGraphError(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)

#Граф состояния
class StateGraph:
	def __init__(self):
		self.nodes = {}
		self.edges= {}
		self.states={}
		self.states_number = 0
		self.g = graph.Graph()

	def add_node(self, value):
		self.nodes[value] = 1
		#self._change_number_states(state)

	def addStates(self, listofstates):
		'''
			List of states is a function
		'''
		if len(listofstates) == 0:
			raise StateGraphError("List of states is empty. Try to append any state")

		self.states = {i: state for i, state in enumerate(listofstates)}

	def _change_number_states(self, state):
		self.states_number = self.states_number = \
		state if state > self.states_number else self.states_number 

	def get_number_states(self):
		states = self.states
		return 0 if len(states) == 0 else list(states.keys())[-1] 

	#Get node of states
	def get_node(self, node):
		result = []
		for key, state in self.states.items():
			resultfunc = state(*node)
			if resultfunc != '':
				result.append(resultfunc)
		return result

	def add_edge(self, inode, onode):
		#Get random state
		from random import randint
		rstate = randint(0, self.states_number)



