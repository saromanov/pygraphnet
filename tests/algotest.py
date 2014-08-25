import graph
import algorithms
import unittest

class VoitingTest(unittest.TestCase):
	def schulze_method(self):
		matr = numpy.array([[-1,70,33],[27,-1,60], [64,35,-1]])
    	a = Algorithms([])
    	print(a.Schulze_voiting(matr))
    	a.sch_voiting_alt(matr)

    def schulze_method2(self):
    	matr = numpy.array([[-1,11,20,14], [19,-1,9,12], [10,21,-1,17], [16,18,13,-1]])
    	a = Algorithms([])
    	print(a.Schulze_voiting(matr))
    	a.sch_voiting_alt(matr)


class ShortestPath(unittest.TestCase):
    def levit(self):
        lev = algorithms.Levit()
        lev.fit('a')
