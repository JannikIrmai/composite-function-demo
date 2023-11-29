import unittest
import functions


class TestMLP(unittest.TestCase):

    def test_empty(self):
        f = functions.MLP({})
        output = f.forward()
        self.assertEqual(output, 0.5)


if __name__ == "__main__":
    unittest.main()
