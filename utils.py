import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_nips_data():
    """Return a list of abstract and a list of paper content from NIPS conference."""
    path = 'data/nips-papers'
    papers = pd.read_csv(os.path.join(path, 'papers.csv'))
    papers['abstract'] = papers['abstract'].str.replace('\r', '').str.replace('\n', ' ')
    papers['paper_text'] = papers['paper_text'].str.replace('\n', ' ')
    mask = papers['abstract'] != 'Abstract Missing'
    paper_abstract = papers[mask]['abstract'].tolist()
    paper_text = papers[mask]['paper_text'].tolist()  # content of paper with abstract

    return paper_abstract, paper_text


def get_wiki_data():
    """Return a list of summary and a list of WikiHow article."""
    path = 'data'
    wiki = pd.read_csv(os.path.join(path, 'wikihowAll.csv'))
    wiki['headline'] = wiki['headline'].str.replace('\n', ' ')
    wiki['text'] = wiki['text'].str.replace('\n', ' ')

    return wiki['headline'].tolist(), wiki['text'].tolist()


def vectorize_text(doc, split = '. '):
    """Vecotrize the sentences in the document based on Tfidf.

    Argument:
        - doc: document to be vectorized (string)
        - split: delineator used to split sentences
    """
    corpus = doc.split(split)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)  # (num_sentence, num_vocab)

    return X


def print_summary(doc, exemplar_indices, split='. '):
    corpus = doc.split(split)
    print('=============== Extracted summary ===============')
    for idx in exemplar_indices:
        print(corpus[idx])
        print('\n')
