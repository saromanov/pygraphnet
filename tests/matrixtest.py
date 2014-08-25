import inittest
import Matrix
import unittest

class TestAdjacency(unittest.TestCase):
	def setUp(self):
		self.adg = Matrix.Adjacency()

	def test_basic(self):
		self.adg.add_node('value')
		self.adg.add_node('some')
		self.adg.add_node('data')
		self.adg.add_edge('value', 'some')
		self.adg.add_node('doom')
		self.adg.add_edge('doom', 'value')
		self.assertEqual(self.adg.data.shape, (4,4))
		#self.assertEquals(self.adg.data[0], [0,1,0,0])
if __name__ == '__main__':
	unittest.main()