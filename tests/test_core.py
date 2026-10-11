import unittest

from worker_vector import SuccinctBitVector


class TestEmpty(unittest.TestCase):
    def test_empty_rank(self):
        sbv = SuccinctBitVector([])
        self.assertEqual(len(sbv), 0)
        self.assertEqual(sbv.rank1(0), 0)
        self.assertEqual(sbv.rank0(0), 0)

    def test_empty_select_raises(self):
        sbv = SuccinctBitVector([])
        with self.assertRaises(ValueError):
            sbv.select1(0)
        with self.assertRaises(ValueError):
            sbv.select0(0)


if __name__ == "__main__":
    unittest.main()
