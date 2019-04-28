import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

import utils
from frank_wolfe import DR_Frank_Wolfe


def extract_summary(doc, ref=None, title=None, k=5, print_summary=False, report_rouge=False, print_rouge=True,
                    rouge_embed=False, vectorize_scores=False, methods=['random', 'SMRS', 'tfidf', 'embed']):
    X = utils.vectorize_text(doc)  # tfidf vectors
    if print_summary: print("# sentence: %d, # vocab: %d"  %(X.shape[0], X.shape[1]))  # (num_sentence, num_vocab)
    num_ref = len(utils.split_sentence(ref))
    if num_ref < k: k = num_ref  # update k if num of sentences in reference is smaller
    if print_summary: print('# of selected exemplar: %d' %k)
    if print_summary and title: print('\nTitle: ' + title + '\n')
    if print_summary and ref:
        print('=============== Referecne Text ==============')
        print(ref)

    summarizer = DR_Frank_Wolfe(epsilon = 0, beta = 10, zeta = 0, positive = False,
                                greedy=True, order = 2, do_centering = False,
                                do_add_equality_constraint = True, num_exemp = k, verbose = True)
    summary, runtime = {}, {}
    if 'random' in methods:
        if print_summary: print('\n========== Extracted summary: random selection ==========')
        start = time.time()
        random_exemplar_indices = [np.random.randint(X.shape[0]) for _ in range(k)]
        summary['random'] = utils.get_summary(doc, random_exemplar_indices, print_summary)
        runtime['random'] = time.time() - start
        if print_summary: print('Random selection computation time: %.3f' %(runtime['random']))

    if 'SMRS' in methods:
        if print_summary: print('\n========== Extracted summary: SMRS ==========')
        eng = utils.start_matlab_engine()
        start = time.time()
        SMRS_exemplar_indices = np.asarray(eng.smrs(utils.convert_csr_matrix_to_matlab_mat(X.T), 5, 0, True)[0])
        SMRS_exemplar_indices = [int(x)-1 for x in SMRS_exemplar_indices][:k]
        summary['SMRS'] = utils.get_summary(doc, SMRS_exemplar_indices, print_summary)
        runtime['SMRS'] = time.time() - start
        if print_summary: print('SMRS computation time: %.3f' %(runtime['SMRS']))

    if 'tfidf' in methods:
        if print_summary: print('\n========== Extracted summary: Tfidf ==========')
        start = time.time()
        tfidf_exemplar_indices, _ = summarizer.identify_exemplars(X)
        summary['tfidf'] = utils.get_summary(doc, tfidf_exemplar_indices, print_summary)
        runtime['tfidf'] = time.time() - start
        if print_summary: print('Tfidf computation time: %.3f' %(runtime['tfidf']))

    if 'embed' in methods:
        # Sentence embeddings
        if print_summary: print('\n====== Extracted summary: sentence embeddings ======')
        #glove_dict = utils.get_glove_dict()
        embed = utils.embed_sentence(doc)
        #if print_summary: print('Sentence embedding shape: (%d, %d)' %(embed.shape[0], embed.shape[1]))
        start = time.time()
        embed_exemplar_indices, _ = summarizer.identify_exemplars(embed)
        summary['embed'] = utils.get_summary(doc, embed_exemplar_indices, print_summary)
        runtime['embed'] = time.time() - start
        if print_summary: print('Sentence embedding computation time: %.3f' %(runtime['embed']))

    # Report ROUGE scores
    if report_rouge and ref:
        scores = {}
        if print_rouge: print('\n=============== ROUGE Scores ===============')
        for method in methods:
            if print_rouge: print('\n' + method)
            if rouge_embed:
                scores[method] = utils.get_rouge_score(summary[method], ref, print_rouge, vectorize_scores, rouge_embed=True)
            else:
                scores[method] = utils.get_rouge_score(summary[method], ref, print_rouge, vectorize_scores)

        return summary, runtime, scores

    return summary, runtime


def report_rouge_scores(docs, refs, titles=None, k=5, rouge_embed=False, methods=['random', 'SMRS', 'tfidf', 'embed']):
    rouge_scores = defaultdict(list) # dictionary to store all rouge scores
    for doc, ref, title in zip(docs, refs, titles):
        _, _, scores = extract_summary(doc, ref, title, k, print_summary=False, report_rouge=True, print_rouge=False,
                                        rouge_embed=rouge_embed, vectorize_scores=True, methods=methods)  # summary text won't be printed
        for key in scores.keys(): rouge_scores[key].append(scores[key])
    rouge_mean, rouge_median, rouge_std = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for key in rouge_scores.keys():
        rouge_mean[key] = np.mean(rouge_scores[key], axis=0)
        rouge_median[key] = np.median(rouge_scores[key], axis=0)
        rouge_std[key] = np.std(rouge_scores[key], axis=0)
    if rouge_embed:
        return rouge_mean[:-3], rouge_median[:-3], rouge_std[:-3]
    return rouge_mean, rouge_median, rouge_std
