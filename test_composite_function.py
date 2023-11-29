import unittest

import numpy as np

from composite_function import CompositeFunction


class TestCompositeFunction(unittest.TestCase):

    def test_default(self):
        f = CompositeFunction()
        output = f.inference()
        self.assertEqual([], output)

    def test_add_nodes_and_links(self):
        f = CompositeFunction()
        n_input = f.add_input()
        n_interior = f.add_interior_node()
        n_output = f.add_output()
        f.add_link(n_input, n_output)
        f.add_link(n_interior, n_output)
        output = f.inference([0.0, 1.0, 2.0])
        self.assertTrue(np.array_equal(np.array([[0.0, 1.0, 2.0]]), output))


if __name__ == "__main__":
    unittest.main()
