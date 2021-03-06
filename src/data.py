import os
import io
import json
import pandas as pd
from collections import defaultdict
import xml.etree.ElementTree as ET

import utils


def get_nips_data(num=-1):
    """Return a list of abstract and a list of paper content from NIPS conference."""
    path = '../data/nips-papers'
    papers = pd.read_csv(os.path.join(path, 'papers.csv'))
    papers['abstract'] = papers['abstract'].str.replace('\r', '').str.replace('\n', ' ')
    papers['paper_text'] = papers['paper_text'].str.replace('\n', ' ')
    mask = papers['abstract'] != 'Abstract Missing'
    paper_title = papers[mask]['title'].tolist()
    paper_abstract = papers[mask]['abstract'].tolist()
    paper_text = papers[mask]['paper_text'].tolist()  # content of paper with abstract

    return paper_title[:num], paper_abstract[:num], paper_text[:num]


def get_wiki_data(num=-1):
    """Return a list of summary and a list of WikiHow article."""
    path = '../data'
    wiki = pd.read_csv(os.path.join(path, 'wikihowAll.csv'))
    wiki['headline'] = wiki['headline'].str.replace('\n', ' ').str.replace(',', '')
    wiki['text'] = wiki['text'].str.replace('\n', ' ')

    return wiki['title'].tolist()[:num], wiki['headline'].tolist()[:num], wiki['text'].tolist()[:num]


def get_outlook_data(name=None, num=None):
    if name == 'citi':
        data = pd.read_csv('../data/2019-outlooks/citi.csv')
    elif name == 'cs':
        data = pd.read_csv('../data/2019-outlooks/cs.csv')
    elif name == 'rbc':
        data = pd.read_csv('../data/2019-outlooks/rbc.csv')
    else:
        data1 = pd.read_csv('../data/2019-outlooks/citi.csv')
        data2 = pd.read_csv('../data/2019-outlooks/cs.csv')
        data3 = pd.read_csv('../data/2019-outlooks/rbc.csv')
        data = pd.concat([data1, data2, data3], sort=False, ignore_index=True)
        if num:
            data = data.sample(n=num)
        data['text'] = data['text'].apply(clean_text)

    return data['title'].tolist(), data['reference'].tolist(), data['text'].tolist()


def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\xe2\x80\x93', '-')
    text = text.replace('\xe2\x80\x99', "'")
    text = text.replace('\xe2\x80\x98', "''")
    text = text.replace('\x97', '')

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


def correct_legal_case_xml():
    path = "../data/legal-case/fulltext/"
    all_files = os.listdir(path)
    for file_name in all_files:
        with open(path + file_name, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if line[:13] == "<catchphrase ":
                    stop = 20 if line[20] == ">" else 21
                    newline = line[:13] + 'id="' + line[17:stop-1] + line[stop-1:]
                    lines[idx] = newline

        with open(path + file_name, 'w') as file:
            for line in lines:
                file.write(line)


def legal_case_xml2csv():
    path = "../data/legal-case/fulltext/"
    all_files = listdir(path)
    titles, references, text = [], [], []
    error = 0
    for file_name in all_files:
        try:
            tree = ET.parse(path + file_name)
            root = tree.getroot()
            titles.append(root.find('name').text)

            ref = '. '.join([phrase.text for phrase in root.findall('./catchphrases/catchphrase')])
            references.append(ref)

            sentences = '. '.join([phrase.text for phrase in root.findall('./sentences/sentence')]).replace('\n', '').replace("\'", '')
            text.append(sentences)
        except:
            error += 1

    print('# error document', error)
    legal_case_data = pd.DataFrame({'reference': references, 'title': titles, 'text': text})
    legal_case_data.to_csv('../data/legal-case/legal_case.csv', index=False, encoding='utf-8')

    return legal_case_data


def get_legal_case_data():
    data = pd.read_csv('../data/legal-case/legal_case.csv')
    mask = data['reference'].apply(utils.split_sentence).apply(len) > 5
    data = data[mask]
    return data['title'].tolist(), data['reference'].tolist(), data['text'].tolist()


def newsroom_json2csv():
    datastore = []
    with open('../data/newsroom/dev.jsonl', encoding='utf-8') as file:
        for f in file:
            data = json.loads(f)
            if (data['density_bin'] == 'extractive'):# and (len(data['summary'].split('. ')) >= 5):
            #if len(data['summary'].split('. ')) >= 5:
                datastore.append(data)

    titles, references, text = [], [], []
    for data in datastore:
        titles.append(data['title'])
        references.append(data['summary'])
        text.append(data['text'])

    data = pd.DataFrame()
    data['title'] = titles
    data['reference'] = references
    data['text'] = text
    data.to_csv('../data/newsroom/news_dev.csv', index=False, encoding='utf-8')


def get_newsroom_data(num=-1):
    data = pd.read_csv('../data/newsroom/news_dev.csv')
    #data['reference'] = data['reference'].apply(clean_text)
    data['text'] = data['text'].apply(clean_text)
    return data['title'].tolist()[:num], data['reference'].tolist()[:num], data['text'].tolist()[:num]


def get_reddit_data(num=-1):
    data = pd.read_csv('../data/aligned-summarization-data/reddit.csv')
    data['text'] = data['text'].apply(clean_text)
    return None, data['reference'].tolist()[:num], data['text'].tolist()[:num]


def get_cnn_data(num=None):
    data = pd.read_csv('../data/cnn.csv')
    data['text'] = data['text'].apply(clean_text)
    if num:
        data = data.sample(n=num)
        return None, data['reference'].tolist(), data['text'].tolist()
    return None, data['reference'].tolist(), data['text'].tolist()


def get_dailymail_data(num=None):
    data = pd.read_csv('../data/dailymail.csv')
    data['text'] = data['text'].apply(clean_text)
    if num:
        data = data.sample(n=num)
        return None, data['reference'].tolist(), data['text'].tolist()
    return None, data['reference'].tolist(), data['text'].tolist()


def get_classical_book_data(num=None):
    data = pd.read_csv('../data/classical_books.csv')
    data['text'] = data['text'].apply(clean_text)
    if num:
        data = data.sample(n=num)
        return None, data['reference'].tolist(), data['text'].tolist()
    return data['title'].tolist(), data['reference'].tolist(), data['text'].tolist()
