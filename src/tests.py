import unittest

import numpy as np
import scipy

import data
import utils
from frank_wolfe import DR_Frank_Wolfe

class TestExtractSum(unittest.TestCase):

    def setUp(self):
        """Set up data for the project."""
        self.titles, self.refs, self.docs = data.get_outlook_data()
        self.doc = self.docs[0]

    def test_get_outlook_data(self):
        num_titles, num_refs, num_docs = len(self.titles), len(self.refs), len(self.docs)
        self.assertTrue(num_titles > 0)
        self.assertTrue(num_titles == num_refs == num_docs)


if __name__ == '__main__':
    unittest.main()
