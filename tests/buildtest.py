import inittest
import graph
import builder
import unittest


class TreeBuilder(unittest.TestCase):
	def test_binary_tree(self):
		tree = builder.BinaryTree('a')
		tree.addNodes('a', 'b', 'c')
		tree.addNodes('b', 'e', 'f')
		#tree.addNodes('e', 'y', 'a')
		self.assertEqual(tree.checkNode('a'), True)
		self.assertEqual(tree.checkNode('q'), False)
		self.assertEqual(tree.checkNode('e'), False)

	def test_binary_tree_with_levels(self):
		tree = builder.BinaryTree('a', levels=3)
		tree.addNodes('a', 'd', 'c')
		tree.addNodes('d', 'ba', 'w')
		tree.addNodes('q', 'aa', 'ee')


if __name__ == "__main__":
    unittest.main()