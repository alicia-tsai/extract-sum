import re
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

from rouge_scores import Rouge


def split_sentence(doc, split = "\.\s|\?\s|\!\s"):
    """Split sentences of the given documents."""
    return re.split(split, doc)


def vectorize_text(doc):
    """Vecotrize the sentences in the document based on Tfidf.

    Argument:
        - doc: document to be vectorized (string)
        - split: delineator used to split sentences (regular expression)
    """
    corpus = split_sentence(doc)
    vectorizer = TfidfVectorizer(encoding='utf-8')
    X = vectorizer.fit_transform(corpus)  # (num_sentence, num_vocab)

    return X


def get_summary(doc, exemplar_indices, verbose=True):
    corpus = split_sentence(doc)
    summary = '. '.join([corpus[idx].capitalize() for idx in exemplar_indices])
    if verbose: print(summary)

    return summary


def get_glove_dict():
    """Return Glove word embeddings trained from Twitter text."""
    glove_dict = {}
    with open('../embeddings/glove.twitter.27B.25d.txt', 'r') as f:
        for line in f.readlines():
            glove_dict[line.split()[0]] = np.array(line.split()[1:], dtype=np.float32)

    return glove_dict


def embed_sentence(doc, word_vectors='en_core_web_lg'):
    """Return sentence embeddings (average word embeddings)."""
    corpus = split_sentence(doc)
    nlp = spacy.load(word_vectors)

    embeddings = None
    for sentence in corpus:
        #vector = [glove_dict[word] for word in sentence.split() if word in word_vectors]
        tokens = nlp(sentence)
        vector = [token.vector for token in tokens if token.has_vector]
        if embeddings is None:
            embeddings = np.mean(vector, axis=0)
        else:
            #if np.mean(vector, axis=0).shape != embeddings.shape:
            if len(vector) == 0:
                continue  # skip if all words in the sentences are not in the embeddings
            embeddings = np.vstack((embeddings, np.mean(vector, axis=0)))

    #embeddings = np.array(embeddings)
    #embeddings_sparse = sparse.csr_matrix(embeddings)

    return embeddings#, embeddings_sparse


def get_rouge_score(summary, reference, verbose=True, vectorize=False, rouge_embed=False, embed_dict=None):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference, avg=True, rouge_embed=rouge_embed, embed_dict=embed_dict)
    if verbose:
        print('Overlap 1-gram \t\t\tF1: %.3f' %(scores['rouge-1']['f']))
        print('Overlap 1-gram \t\t\tPrecision: %.3f' %(scores['rouge-1']['p']))
        print('Overlap 1-gram \t\t\tRecall: %.3f' %(scores['rouge-1']['r']))

        print('Overlap bi-gram \t\tF1: %.3f' %(scores['rouge-2']['f']))
        print('Overlap bi-gram \t\tPrecision: %.3f' %(scores['rouge-2']['p']))
        print('Overlap bi-gram \t\tRecall: %.3f' %(scores['rouge-2']['r']))

        if not rouge_embed:  # LCS embed not implemented
            print('Longest Common Subsequence \tF1: %.3f' %(scores['rouge-l']['f']))
            print('Longest Common Subsequence \tPrecision: %.3f' %(scores['rouge-l']['p']))
            print('Longest Common Subsequence \tRecall: %.3f' %(scores['rouge-l']['r']))
    if vectorize:
        # return scores in a vector
        return np.array([v for name in scores.values() for v in name.values()])
    return scores


def start_matlab_engine():
    """Enable matlab engine."""
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd("../SMRS_v1.0")
        return eng
    except ImportError:
        print("Matlab not imported")


def convert_csr_matrix_to_matlab_mat(A):
    try:
        import matlab.engine
        return matlab.double(A.toarray().tolist(), A.shape)
    except ImportError:
        print("Matlab not imported")
