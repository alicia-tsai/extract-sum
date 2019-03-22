import os
import io
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge


def get_nips_data():
    """Return a list of abstract and a list of paper content from NIPS conference."""
    path = 'data/nips-papers'
    papers = pd.read_csv(os.path.join(path, 'papers.csv'))
    papers['abstract'] = papers['abstract'].str.replace('\r', '').str.replace('\n', ' ')
    papers['paper_text'] = papers['paper_text'].str.replace('\n', ' ')
    mask = papers['abstract'] != 'Abstract Missing'
    paper_title = papers[mask]['title'].tolist()
    paper_abstract = papers[mask]['abstract'].tolist()
    paper_text = papers[mask]['paper_text'].tolist()  # content of paper with abstract

    return paper_title, paper_abstract, paper_text


def get_wiki_data():
    """Return a list of summary and a list of WikiHow article."""
    path = 'data'
    wiki = pd.read_csv(os.path.join(path, 'wikihowAll.csv'))
    wiki['headline'] = wiki['headline'].str.replace('\n', ' ')
    wiki['text'] = wiki['text'].str.replace('\n', ' ')

    return wiki['title'].tolist(), wiki['headline'].tolist(), wiki['text'].tolist()


def get_outlook_data():
    citi = pd.read_csv('data/2019-outlooks/citi.csv')
    citi_title = citi['title'].apply(clean_text)
    citi_reference = citi['reference'].apply(clean_text)
    citi_text = citi['text'].str.lower().apply(clean_text)

    return citi_title, citi_reference.tolist(), citi_text.tolist()


def clean_text(text):
    text = text.replace('\xe2\x80\x93', '-')
    text = text.replace('\xe2\x80\x99', "'")
    text = text.replace('\xe2\x80\x98', "''")

    return text


def get_bible_data():
    with io.open('data/bible/bible.txt', 'r', encoding='utf-8') as file:
        bible_raw = []
        for line in file.readlines():
            bible_raw.append(line.replace('\r', '').replace('\n', ''))
        bible = defaultdict(list)
        for text in bible_raw:
            bible[text.split()[0]].append(' '.join(text.split()[2:]))

    return bible


def split_sentence(doc, split = "\.\s|\?\s|\!\s"):
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


def get_summary(doc, exemplar_indices, print_flag=True):
    corpus = split_sentence(doc)
    summary = '. '.join([corpus[idx].capitalize() for idx in exemplar_indices])
    if print_flag:
        print(summary)

    return summary


def get_glove_dict():
    """Return Glove word embeddings trained from Twitter text."""
    glove_dict = {}
    with open('embeddings/glove.twitter.27B.25d.txt', 'r') as f:
        for line in f.readlines():
            glove_dict[line.split()[0]] = np.array(line.split()[1:], dtype=np.float32)

    return glove_dict


def embed_sentence(doc, glove_dict):
    """Return sentence embeddings (average word embeddings)."""
    corpus = split_sentence(doc)
    embeddings = None
    for sentence in corpus:
        vector = [glove_dict[word] for word in sentence.split() if word in glove_dict]
        if embeddings is None:
            embeddings = np.mean(vector, axis=0)
        else:
            if np.mean(vector, axis=0).shape != embeddings.shape
            np.vstack((embeddings, np.mean(vector, axis=0)))

    #embeddings = np.array(embeddings)
    #embeddings_sparse = sparse.csr_matrix(embeddings)

    return embeddings#, embeddings_sparse


def get_rouge_score(summary, reference, print_flag=True):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference, avg=True)
    if print_flag:
        print('Overlap 1-gram \t\t\tF1: %.3f' %(scores['rouge-1']['f']))
        print('Overlap 1-gram \t\t\tPrecision: %.3f' %(scores['rouge-1']['p']))
        print('Overlap 1-gram \t\t\tRecall: %.3f' %(scores['rouge-1']['r']))

        print('Overlap bi-gram \t\tF1: %.3f' %(scores['rouge-2']['f']))
        print('Overlap bi-gram \t\tPrecision: %.3f' %(scores['rouge-2']['p']))
        print('Overlap bi-gram \t\tRecall: %.3f' %(scores['rouge-2']['r']))

        print('Longest Common Subsequence \tF1: %.3f' %(scores['rouge-l']['f']))
        print('Longest Common Subsequence \tPrecision: %.3f' %(scores['rouge-l']['p']))
        print('Longest Common Subsequence \tRecall: %.3f' %(scores['rouge-l']['r']))
    return scores
