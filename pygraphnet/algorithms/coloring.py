import itertools
import sys
sys.path.append('..')

import graph

#Implementation of Graph coloring algorithms

class Coloring:
	def __init__(self, sgraph):
		self.sgraph = sgraph
		self.edges = list(self.sgraph.get_edges())
		self.nodes = list(map(lambda x:x.node, self.sgraph.nodes()))

	def _assertGraph(self):
		if len(self.nodes) == 1:
			return True,1
		if len(self.nodes) == 2:
			return True,2
		return False,-1


class ChromaticNumber(Coloring):
    def __init__(self, sgraph):
    	Coloring.__init__(self, sgraph)

    def _checkNeighboring(self, colors, data, node):
        return 1 if len(list(itertools.dropwhile(lambda x: node not in data[x], \
        	colors.keys()))) == 0 else 0

    def run(self):
        '''
            Naive approach,
            return - min number of colors
            Кликовое число графа
            Welsh-Powell Algorithm
        '''
        if self._isFullGraph():
            return len(sgraph.nodes())
        resp, num = self._assertGraph()
        if resp: return num

        sortedges = list(reversed(sorted(self.edges, \
            key=lambda x: len(x[1]))))
        data = {i:j for i,j in sortedges}
        colors = {}
        color = 1
        while len(nodes) > 0:
        	colors[sortedges[0][0]] = color
        	newnodes = self._innerLoop(sortedges, colors, data)
        	for ne in newnodes:
        		colors[ne] = color
        		nodes.remove(ne)
        	sortedges.remove((sortedges[0][0], sortedges[0][1]))
        	color += 1

        return colors

    def _innerLoop(self, sortededges, colors, data):
    	return list(map(lambda x: x[0], \
    		filter(lambda x: self._checkNeighboring(colors, data, x[0]), sortededges)))

    def _isFullGraph(self):
        return len(list(filter(lambda x: len(x[1]) != len(self.sgraph.nodes())-1, self.edges))) == 0



class HarmoniousColoring(Coloring):
	def __init__(self, sgraph):
		Coloring.__init__(self, sgraph)

	def run(self):
		resp, num = self._assertGraph()
		if resp: return num

