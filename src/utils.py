import sys
sys.path.append('../embeddings/SIF/src')
import re
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.linalg import sqrtm
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.summarization.bm25 import get_bm25_weights

from rouge_scores import Rouge
import skipthoughts
import data_io, params, SIF_embedding

from gensim.summarization.textcleaner import clean_text_by_sentences
# from gensim.summarization.summarizer import _build_corpus, _build_hasheable_corpus, _set_graph_edge_weights
# from gensim.summarization.commons import build_graph, remove_unreachable_nodes
# from gensim.summarization.pagerank_weighted import build_adjacency_matrix


def split_sentence(doc):
    """Split sentences of the given documents."""
    return sent_tokenize(doc)


def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def clean_sentences(sentences, clean_stopwords=True):
    # remove punctuations, numbers and special characters
    new_sentences = [re.sub("[^a-zA-Z\s]", "", s) for s in sentences]
    # remove stopwords from the sentences
    if clean_stopwords:
        new_sentences = [remove_stopwords(s.split()) for s in new_sentences]
    return new_sentences

def vectorize_text(doc):
    """Vecotrize the sentences in the document based on Tfidf.

    Argument:
        - doc: document to be vectorized (string)
    """
    #corpus = split_sentence(doc)
    #corpus = clean_sentences(corpus)
    corpus = [sen.token for sen in clean_text_by_sentences(doc)]
    vectorizer = TfidfVectorizer(encoding='utf-8')
    return vectorizer.fit_transform(corpus)  # (num_sentence, num_vocab)

def get_bm25_kernel(doc):
    #corpus = split_sentence(doc)
    #corpus = clean_sentences(corpus)
    corpus = [sen.token for sen in clean_text_by_sentences(doc)]
    corpus = [s.split() for s in corpus]
    X = sqrtm(np.array(get_bm25_weights(corpus, n_jobs=-1)))
    #X[0:3] += 1

    # hashable_corpus = _build_hasheable_corpus(_build_corpus(clean_text_by_sentences(doc)))
    #
    # graph = build_graph(hashable_corpus)
    # _set_graph_edge_weights(graph)
    # #remove_unreachable_nodes(graph)
    # coeff_adjacency_matrix = build_adjacency_matrix(graph)
    # matrix = coeff_adjacency_matrix.toarray()
    # damping=0.85
    # probabilities = (1 - damping) / float(len(graph))
    # matrix += probabilities
    # X = matrix
    return X


def get_summary(doc, exemplar_indices, verbose=True):
    #corpus = split_sentence(doc)
    corpus = [sen.text for sen in clean_text_by_sentences(doc)]
    sentences = [corpus[idx].capitalize() for idx in exemplar_indices]
    word_count = np.sum([len(sen.split()) for sen in sentences])
    summary = ' '.join(sentences)
    if verbose:
        print('\n'.join(sentences))
        print('-' * 5)
        print('Word count:' + str(word_count))

    return summary, word_count


def get_glove_dict():
    """Return Glove word embeddings trained from Twitter text."""
    glove_dict = {}
    with open('../embeddings/glove.twitter.27B.25d.txt', 'r') as f:
        for line in f.readlines():
            glove_dict[line.split()[0]] = np.array(line.split()[1:], dtype=np.float32)

    return glove_dict


def get_skip_thoughts_encoder():
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    return encoder


def embed_sentence(doc, word_vectors='en_core_web_lg', encoder=None, word2weight=None):
    """Return sentence embeddings."""
    # embed sentences with skip thoughs model
    if word_vectors == 'skip-thoughts':
        return embed_sentence_skip_thoughts(doc, encoder)
    #corpus = split_sentence(doc)
    #corpus = clean_sentences(corpus, clean_stopwords=True)
    corpus = [sen.token for sen in clean_text_by_sentences(doc)]

    # embed sentences with Spacy model using average weight
    nlp = spacy.load(word_vectors)
    embeddings = None
    oov = {}
    if word2weight:
        for sentence in corpus:
            if sentence == '':
                continue
            tokens = nlp(sentence)
            vector = []
            weights = 0
            for idx, token in enumerate(tokens):
                if token.has_vector:
                    if token.text in word2weight:
                        #vector.append(token.vector / token.vector_norm * word2weight[token.text])
                        vector.append(token.vector * word2weight[token.text])
                        weights += word2weight[token.text]
                    else:
                        #vector.append(token.vector / token.vector_norm * 6)
                        vector.append(token.vector * 6)
                        weights += 6  # 6 is the lowest frequency in vocab.txt
                # Assign random vector for oov words
                # else:
                #     if token.text not in oov:
                #         # if (idx > 0) and (idx < len(tokens)-1) and (tokens[idx-1].has_vector) and (tokens[idx+1].has_vector):
                #         #     oov[token.text] = (tokens[idx-1].vector + tokens[idx+1].vector) / 2
                #         # else:
                #         oov[token.text] = np.random.random(300)
                #     if token.text in word2weight:
                #         vector.append(oov[token.text] * word2weight[token.text])
                #         weights += word2weight[token.text]
                #     else:
                #         vector.append(token.vector * 6)
                #         weights += 6
            if embeddings is None:
                vec = np.sum(vector, axis=0) / weights
                embeddings = vec / np.linalg.norm(vec)
            else:
                if len(vector) == 0:
                    continue  # skip if all words in the sentences are not in the embeddings
                try:
                    vec = np.sum(vector, axis=0) / weights
                    embeddings = np.vstack((embeddings, vec / np.linalg.norm(vec)))
                except:
                    pass
        # adjust embedding with SIF method
        rmpc = 1 # number of principal components to remove in SIF weighting scheme
        parameters = params.params()
        parameters.rmpc = rmpc
        embeddings = SIF_embedding.SIF_embedding(embeddings, parameters) # embedding[i,:] is the embedding for sentence i
        #embeddings -= np.min(embeddings)
        #embeddings = np.sqrt(embeddings)
        #embeddings[0:3] += 1

        return embeddings

    for sentence in corpus:
        #vector = [glove_dict[word] for word in sentence.split() if word in word_vectors]
        if sentence == '':
            continue
        tokens = nlp(sentence)
        vector = [token.vector for token in tokens if token.has_vector]
        if embeddings is None:
            embeddings = np.mean(vector, axis=0)
        else:
            #if np.mean(vector, axis=0).shape != embeddings.shape:
            if len(vector) == 0:
                continue  # skip if all words in the sentences are not in the embeddings
            embeddings = np.vstack((embeddings, np.mean(vector, axis=0)))
        embeddings = np.sqrt(embeddings)
        embeddings[0] += 1

    return embeddings


def embed_sentence_skip_thoughts(doc, encoder):
    corpus = split_sentence(doc)
    embeddings = encoder.encode(corpus, verbose=False)
    return embeddings


def get_word_weight():
    #weightfile = '../embeddings/SIF/auxiliary_data/vocab.txt' # each line is a word and its frequency
    weightfile = '../embeddings/SIF/auxiliary_data/enwiki_vocab_min200.txt'
    weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
    return word2weight

# def get_SIF_input():
#     # input
#     #wordfile = '../data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website
#     weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
#     weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
#     # load word vectors
#     #(words, We) = data_io.getWordmap(wordfile)
#     # load word weights
#     word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
#     #weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
#     return word2weight


# def embed_sentence_SIF(sentences, SIF_input):
#     word2weight = SIF_input
#     rmpc = 1 # number of principal components to remove in SIF weighting scheme
#     # load sentences
#     x, m, _ = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
#     w = data_io.seq2weight(x, m, weight4ind) # get word weights
#     # set parameters
#     params = params.params()
#     params.rmpc = rmpc
#     # get SIF embedding
#     embeddings = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
#     return embeddings


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
        if type(A) == np.ndarray:
            return matlab.double(A.tolist(), A.shape)
        return matlab.double(A.toarray().tolist(), A.shape)
    except ImportError:
        print("Matlab not imported")


def diversity_score(summary, word_vectors='en_core_web_lg', word2weight=None):
    if len(summary) == 0:
        return 0
    sent_emb = embed_sentence(summary, word_vectors=word_vectors, word2weight=word2weight)
    print(np.linalg.norm(sent_emb, axis=1))
    return np.sum(np.max(sent_emb.dot(sent_emb.T), axis=0))
