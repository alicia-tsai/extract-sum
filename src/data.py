import os
import io
import pandas as pd
from collections import defaultdict


def get_nips_data():
    """Return a list of abstract and a list of paper content from NIPS conference."""
    path = '../data/nips-papers'
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
    path = '../data'
    wiki = pd.read_csv(os.path.join(path, 'wikihowAll.csv'))
    wiki['headline'] = wiki['headline'].str.replace('\n', ' ')
    wiki['text'] = wiki['text'].str.replace('\n', ' ')

    return wiki['title'].tolist(), wiki['headline'].tolist(), wiki['text'].tolist()


def get_outlook_data():
    citi = pd.read_csv('../data/2019-outlooks/citi.csv')
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
