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

    def test_split_sentence(self):
        sentences = utils.split_sentence(self.doc)
        self.assertEqual(str, type(self.doc))
        self.assertTrue(len(sentences) > 0)
        self.assertEqual(str, type(sentences[0]))

    def test_get_tfidf_matrix(self):
        X = utils.vectorize_text(self.doc)
        self.assertEqual(scipy.sparse.csr.csr_matrix, type(X))
        self.assertTrue(scipy.sparse.linalg.norm(X) > 0)
        self.assertTrue(X.shape[0] > 0)
        self.assertTrue(X.shape[1] > 1)

    def test_get_summary(self):
        k = 5
        X = utils.vectorize_text(self.doc)
        random_exemplar_indices = [np.random.randint(X.shape[0]) for _ in range(k)]
        summary = utils.get_summary(self.doc, random_exemplar_indices, verbose=False)
        self.assertEqual(str, type(summary))

    def test_initialize_franke_wolfe_sparse_representation(self):
        summarizer = DR_Frank_Wolfe(epsilon = 0, beta = 10, zeta = 0, positive = False,
                                    greedy=True, order = 2, do_centering = False,
                                    do_add_equality_constraint = True, num_exemp = 5, verbose = False)
        self.assertEqual(DR_Frank_Wolfe, type(summarizer))


if __name__ == '__main__':
    unittest.main()
