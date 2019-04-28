import unittest

import numpy as np
import scipy

import data
import utils
from frank_wolfe import DR_Frank_Wolfe

class TestExtractSum(unittest.TestCase):

    def setUp(self):
        """Set up data for the project."""
        self.titles, self.refs, self.docs = data.get_newsroom_data()
        self.doc = self.docs[0]



if __name__ == '__main__':
    unittest.main()
