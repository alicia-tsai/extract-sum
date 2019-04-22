import os
import io
import pandas as pd
from collections import defaultdict
import xml.etree.ElementTree as ET


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


def get_outlook_data(name=None):
    if name == 'citi':
        data = pd.read_csv('../data/2019-outlooks/citi.csv')
    else:
        data = pd.read_csv('../data/2019-outlooks/cs.csv')
    data_title = data['title'].apply(clean_text)
    data_reference = data['reference'].apply(clean_text)
    data_text = data['text'].str.lower().apply(clean_text)

    return data_title, data_reference.tolist(), data_text.tolist()


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
    return data['title'].tolist(), data['reference'].tolist(), data['text'].tolist()
