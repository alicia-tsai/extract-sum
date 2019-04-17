import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

import utils
from frank_wolfe import DR_Frank_Wolfe


def extract_summary(doc, ref=None, title=None, k=5, verbose=True, rouge_embed=False, vectorize_scores=False):
    X = utils.vectorize_text(doc)  # tfidf vectors
    if verbose: print("# sentence: %d, # vocab: %d"  %(X.shape[0], X.shape[1]))  # (num_sentence, num_vocab)
    num_ref = len(utils.split_sentence(ref))
    if num_ref < k: k = num_ref  # update k if num of sentences in reference is smaller
    if verbose: print('# of selected exemplar: %d' %k)
    if verbose and title: print('\nTitle: ' + title + '\n')
    if verbose and ref:
        print('=============== Referecne Text ==============')
        print(ref)

    summarizer = DR_Frank_Wolfe(epsilon = 0, beta = 10, zeta = 0, positive = False,
                                greedy=True, order = 2, do_centering = False,
                                do_add_equality_constraint = True, num_exemp = k, verbose = True)
    summary, runtime = {}, {}
    if verbose: print('\n========== Extracted summary: random selection ==========')
    start = time.time()
    random_exemplar_indices = [np.random.randint(X.shape[0]) for _ in range(k)]
    summary['random'] = utils.get_summary(doc, random_exemplar_indices, verbose)
    runtime['random'] = time.time() - start
    if verbose: print('Random selection computation time: %.3f' %(runtime['random']))

    if verbose: print('\n========== Extracted summary: Tfidf ==========')
    start = time.time()
    tfidf_exemplar_indices, _ = summarizer.identify_exemplars(X)
    summary['tfidf'] = utils.get_summary(doc, tfidf_exemplar_indices, verbose)
    runtime['tfidf'] = time.time() - start
    if verbose: print('Tfidf computation time: %.3f' %(runtime['tfidf']))

    # Sentence embeddings
    if verbose: print('\n====== Extracted summary: sentence embeddings ======')
    glove_dict = utils.get_glove_dict()
    embed = utils.embed_sentence(doc, glove_dict)
    if verbose: print('Sentence embedding shape: (%d, %d)' %(embed.shape[0], embed.shape[1]))
    start = time.time()
    embed_exemplar_indices, _ = summarizer.identify_exemplars(embed)
    summary['embed'] = utils.get_summary(doc, embed_exemplar_indices, verbose)
    runtime['embed'] = time.time() - start
    if verbose: print('Sentence embedding computation time: %.3f' %(runtime['embed']))

    # Report ROUGE scores
    if ref:
        methods = ['random', 'tfidf', 'embed']
        scores = {}
        if verbose: print('\n=============== ROUGE Scores ===============')
        for method in methods:
            if verbose: print('\n' + method)
            if rouge_embed:
                scores[method] = utils.get_rouge_score(summary[method], ref, verbose, vectorize_scores, rouge_embed=True, embed_dict=glove_dict)
            else:
                scores[method] = utils.get_rouge_score(summary[method], ref, verbose, vectorize_scores)

        return summary, runtime, scores

    return summary, runtime


def report_rouge_scores(docs, refs, titles=None, k=5, verbose=False, rouge_embed=False):
    rouge_scores = defaultdict(list) # dictionary to store all rouge scores
    for doc, ref, title in zip(docs, refs, titles):
        _, _, scores = extract_summary(doc, ref, title, k, verbose, rouge_embed, vectorize_scores=True)  # summary text won't be printed
        for key in scores.keys(): rouge_scores[key].append(scores[key])
    rouge_mean, rouge_median, rouge_std = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for key in rouge_scores.keys():
        rouge_mean[key] = np.mean(rouge_scores[key], axis=0)
        rouge_median[key] = np.median(rouge_scores[key], axis=0)
        rouge_std[key] = np.std(rouge_scores[key], axis=0)
    if rouge_embed:
        return rouge_mean[:-3], rouge_median[:-3], rouge_std[:-3]
    return rouge_mean, rouge_median, rouge_std
