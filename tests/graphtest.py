
import inittest
import unittest
import graph
import algorithms
import random

class InitGraph(unittest.TestCase):
    """docstring for InitGraph"""

    def test_open(self):
        gr = graph.Graph()
        grp = graph.Graph({'a':[1,2,3], 'b':[7,8,9]})
        lis1 = ['a', 'b']
        lis2 = ['b', 'c']
        grzip = graph.Graph.Zipped(zip(lis1, lis2))

    def test_addnodes(self):
        gr = graph.Graph()
        gr.add_node(78)
        gr.add_node("SUPERNODE")
        gr.add_node("NEXTNODE")
        def add_test_size():
            for x in range(1000):
                gr.add_node(x)
            return gr.size();

        self.assertEqual(add_test_size(),1002)

    def test_addedge(self):
        gr = graph.Graph()
        gr.add_node('A')
        gr.add_node('B')
        gr.add_edge('A','B')
        self.assertEqual(sorted(list(gr.get_edges())), [('A', ['B']),('B', [])])

    def test_addedge_rev(self):
        gr = graph.Graph()
        gr.add_node('A')
        gr.add_node('B')
        gr.add_edge('A','B', rev=True)
        self.assertEqual(sorted(list(gr.get_edges())), [('A', ['B']),('B', ['A'])])

class ShortPathAlgorithms(unittest.TestCase):
    def test_easy_short_path(self):
        gr = graph.Graph({'s':['u', 'x'], 'u':['v', 'x'], 'v':['y'], 'x':['u', 'v', 'y'], 'y':['s', 'v']})
        gr.set_weight('s','u',5)
        gr.set_weight('s','x',10)
        gr.set_weight('u','v',7)
        gr.set_weight('u','x',4)
        gr.set_weight('v','y',13)
        gr.set_weight('x','u',8)
        gr.set_weight('x','v',6)
        gr.set_weight('x','y',10)
        gr.set_weight('y','s',18)
        gr.set_weight('y','s',11)
        #algo = algorithms.GraphAlgorithms.easy_short_path(gr,'s', 'y')


class RandomGraphTest(unittest.TestCase):
    def test_random_graph(self):
        rand = graph.RandomGraph(20,0.2)
        result = list(dict(rand.getEdges()).keys())
        self.assertEqual(result, list(range(1,21)))

class CartesianProductGraphTest(unittest.TestCase):

    def setUp(self):
        g = graph.Graph()
        g.add_node('A')
        g.add_node('B')
        g.add_node('C')
        g.add_node('D')
        g.add_node('E')
        g.add_edge('A', 'B', rev=True)
        g.add_edge('B', 'C', rev=True)
        g.add_edge('B', 'D', rev=True)
        g.add_edge('C', 'E', rev=True)
        g.add_edge('D', 'E', rev=True)
        tg = graph.Graph()
        tg.add_node('W')
        tg.add_node('P')
        tg.add_node('T')
        tg.add_node('K')
        tg.add_edge('W', 'P', rev=True)
        tg.add_edge('P','T', rev=True)
        tg.add_edge('P','K', rev=True)
        self.g = g
        self.tg = tg

    def test_basic_car_product(self):
        gr1 = graph.Graph()
        gr1.add_edge("A","B", rev=True)
        gr1.add_edge("A","C", rev=True)
        gr2 = graph.Graph()
        gr2.add_edge("P","W", rev=True)
        result = gr1 * gr2
        #self.assertEqual(sorted(list(result.get_edges())), value)
    def test_graph_product(self):
        newgraph = self.g * self.tg

    def test_tensor_product(self):
        newgraph = self.g.tensorProduct(self.tg)


if __name__ == "__main__":
    unittest.main()