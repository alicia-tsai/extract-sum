import time
import numpy as np
import pandas as pd
import utils
from frank_wolfe import DR_Frank_Wolfe
from scipy import sparse


def extract_summary(doc, ref=None, title=None, k=5):
    X = utils.vectorize_text(doc)
    print("# sentence: %d, # vocab: %d"  %(X.shape[0], X.shape[1]))  # (num_sentence, num_vocab)

    k = 5  # number of selected exemplar
    print('# of selected exemplar: %d' %k)
    if title: print('\nTitle: ' + title + '\n')
    if ref:
        print('=============== Referecne Text ==============')
        print(ref)

    summarizer = DR_Frank_Wolfe(epsilon = 0, beta = 10, zeta = 0, positive = False,
                                greedy=True, order = 2, do_centering = False,
                                do_add_equality_constraint = True, num_exemp = k, verbose = True)

    print('\n========== Extracted summary: baseline ==========')
    start = time.time()
    baseline_exemplar_indices, _ = summarizer.identify_exemplars(X)
    baseline_summary = utils.get_summary(doc, baseline_exemplar_indices)
    print('Baseline computation time: %.3f' %(time.time() - start))

    # Sentence embeddings
    print('\n====== Extracted summary: sentence embeddings ======')
    glove_dict = utils.get_glove_dict()
    embed = utils.embed_sentence(doc, glove_dict)
    print(embed)
    print('Sentence embedding shape: (%d, %d)' %(embed.shape[0], embed.shape[1]))
    start = time.time()
    exemplar_indices, _ = summarizer.identify_exemplars(embed)
    summary = utils.get_summary(doc, exemplar_indices)
    print('Sentence embedding computation time: %.3f' %(time.time() - start))

    # Rouge scores
    if ref:
        print('\n=============== ROUGE Scores ===============')
        print('Baseline:')
        utils.get_rouge_score(baseline_summary, ref)
        print('\nSentence embeddings:')
        utils.get_rouge_score(summary, ref)
