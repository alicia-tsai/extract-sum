import unittest

import numpy as np
import scipy

import data
import utils

class TestExtractSum(unittest.TestCase):

    def setUp(self):
        """Set up data for the project."""
        self.titles, self.refs, self.docs = data.get_outlook_data()

    def test_get_outlook_data(self):
        num_titles, num_refs, num_docs = len(self.titles), len(self.refs), len(self.docs)
        self.assertTrue(num_titles > 0)
        self.assertTrue(num_titles == num_refs == num_docs)

    def test_get_tfidf_matrix(self):
        doc = self.docs[0]
        X = utils.vectorize_text(doc)
        self.assertEqual(scipy.sparse.csr.csr_matrix, type(X))
        self.assertTrue(scipy.sparse.linalg.norm(X) > 0)


if __name__ == '__main__':
    unittest.main()
