import inittest
import graph
import builder
import unittest
import labs


class TestStateGraph(unittest.TestCase):
	def setUp(self):
		p = labs.StateGraph()
		p.addStates([lambda x,y: 'FUN' if x > y else '', \
			lambda x,y: 'EQUAL' if x == y else ''])
		p.add_node((1, 1))
		p.add_node((2, 1))
		p.add_node((3, 1))
		p.add_node((1,2))
		p.add_node((3,3))
		self.p = p
	def test_countstates(self):
		self.assertEqual(self.p.get_number_states(), 1)

	def test_resultstates(self):
		self.assertEquals(self.p.get_node((3,3)), ['EQUAL'])


class TestTimeGraph(unittest.TestCase):
	def test_basic(self):
		t = labs.TimeGraph(10)
		t.compute_graph()
		t.start()

if __name__ == '__main__':
	unittest.main()