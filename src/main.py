import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from gensim.summarization.summarizer import summarize

import utils
from frank_wolfe import DR_Frank_Wolfe


def extract_summary(doc, ref=None, title=None, k=None, ratio=None, print_summary=False, report_rouge=False, print_rouge=True,
                    rouge_embed=False, vectorize_scores=False, methods=['TextRank', 'FWSR-BM25', 'FWSR-SIF']):
    X = utils.vectorize_text(doc)
    X_bm = utils.get_bm25_kernel(doc)
    if print_summary: print("Soruce Text: %d sentences, %d distinct vocab"  %(X.shape[0], X.shape[1]))  # (num_sentence, num_vocab)
    num_ref = len(utils.split_sentence(ref))
    num_doc = len(utils.split_sentence(doc))
    if k:
        if k < num_ref: k = num_ref
        ratio = k / num_doc
    elif ratio:
        if int(num_doc * ratio) < num_ref:
            k = num_ref
            ratio = k / num_doc
        else:
            k = int(num_doc * ratio)
    else:
        k = num_ref
        ratio = k / num_doc
    if print_summary: print('# of selected sentences: %d' %k)
    if print_summary and title: print('\nTitle: ' + title + '\n')
    if print_summary and ref:
        print('=============== Referecne Text ==============')
        print(ref + '\n' + '-' * 5)
        print('Word count:' + str(len(ref.split())))
        #print('Similarity score:', utils.diversity_score(ref))
    if X.shape[0] < k:
        return None
    summary, word_count, runtime = {}, {}, {}
    summarizer = DR_Frank_Wolfe(epsilon = 0, beta = 100, zeta = 0, positive = True,
                                greedy=True, order = 2, do_centering = True,
                                do_add_equality_constraint = False, num_exemp = k, verbose = True)
    if 'first-k' in methods:
        if print_summary: print('\n========== Extracted summary: First k ==========')
        start = time.time()
        first_k_exemplar_indices = list(range(k))
        runtime['first-k'] = time.time() - start
        summary['first-k'], word_count['first-k'] = utils.get_summary(doc, first_k_exemplar_indices, print_summary)
        if print_summary: print('First_k selection computation time: %.3f' %(runtime['first-k']))
        #if print_summary: print('Similarity score:', utils.diversity_score(summary['first-k']))

    if 'SMRS' in methods:
        if print_summary: print('\n========== Extracted summary: SMRS ==========')
        eng = utils.start_matlab_engine()
        start = time.time()
        SMRS_exemplar_indices = np.asarray(eng.smrs(utils.convert_csr_matrix_to_matlab_mat(X.T), 5, 0, True)[0])
        runtime['SMRS'] = time.time() - start
        SMRS_exemplar_indices = [int(x)-1 for x in SMRS_exemplar_indices][:k]
        summary['SMRS'], word_count['SMRS'] = utils.get_summary(doc, SMRS_exemplar_indices, print_summary)
        if print_summary: print('SMRS computation time: %.3f' %(runtime['SMRS']))
        #if print_summary: print('Similarity score:', utils.diversity_score(summary['SMRS']))

    if 'TextRank' in methods:
        if print_summary: print('\n========== Extracted summary: TextRank ==========')
        start = time.time()
        summary['TextRank'] = summarize(doc, ratio=ratio)
        runtime['TextRank'] = time.time() - start
        sentences = utils.split_sentence(summary['TextRank'])
        word_count['TextRank'] = np.sum([len(sen.split()) for sen in sentences])
        if print_summary:
            print(summary['TextRank'])
            print('Word count:', word_count['TextRank'])
            print('TextRank computation time: %.3f' %(runtime['TextRank']))
            #print('Diversity score:', utils.diversity_score(summary['TextRank']))

    if 'FWSR-BM25' in methods:
        if print_summary: print('\n========== Extracted summary: FWSR-BM25 ==========')
        start = time.time()
        tfidf_exemplar_indices, _ = summarizer.identify_exemplars(X_bm)
        runtime['FWSR-BM25'] = time.time() - start
        summary['FWSR-BM25'], word_count['FWSR-BM25'] = utils.get_summary(doc, tfidf_exemplar_indices, print_summary)
        if print_summary: print('FWSR-BM25 computation time: %.3f' %(runtime['FWSR-BM25']))
        #if print_summary: print('Similarity score:', utils.diversity_score(summary['tfidf']))

    if 'FWSR-SIF' in methods:
        # Sentence embeddings
        if print_summary: print('\n====== Extracted summary: FWSR-SIF ======')
        #glove_dict = utils.get_glove_dict()
        # skip thoughts
        #encoder = utils.get_skip_thoughts_encoder()
        #embed = utils.embed_sentence(doc, word_vectors='skip-thoughts', encoder=encoder)
        word2weight = utils.get_word_weight()
        embed = utils.embed_sentence(doc, word2weight=word2weight)
        #embed = utils.embed_sentence(doc)
        if print_summary: print('Sentence embedding shape: (%d, %d)' %(embed.shape[0], embed.shape[1]))
        start = time.time()
        embed_exemplar_indices, _ = summarizer.identify_exemplars(embed)
        runtime['FWSR-SIF'] = time.time() - start
        summary['FWSR-SIF'], word_count['FWSR-SIF'] = utils.get_summary(doc, embed_exemplar_indices, print_summary)
        if print_summary: print('FWSR-SIF computation time: %.3f' %(runtime['FWSR-SIF']))
        #if print_summary: print('Similarity score:', utils.diversity_score(summary['embed']))

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

        return summary, word_count, runtime, scores

    return summary, word_count, runtime


def report_rouge_scores(docs, refs, titles=None, k=None, ratio=None, rouge_embed=False, methods=['TextRank', 'FWSR-BM25', 'FWSR-SIF']):
    rouge_scores = defaultdict(list) # dictionary to store all rouge scores
    if not titles: titles = [None] * len(docs)
    for doc, ref, title in zip(docs, refs, titles):
        results = extract_summary(doc, ref, title, k=k, ratio=ratio,
                                        print_summary=False, report_rouge=True, print_rouge=False,
                                        rouge_embed=rouge_embed, vectorize_scores=True, methods=methods)  # summary text won't be printed
        if results is not None:
            summary, word_count, runtime, scores = results
        for method in scores.keys():
            rouge_scores[method].append(np.hstack([scores[method], runtime[method], word_count[method]]))
    rouge_mean, rouge_median, rouge_std = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for method in rouge_scores.keys():
        rouge_mean[method] = np.mean(rouge_scores[method], axis=0)
        rouge_median[method] = np.median(rouge_scores[method], axis=0)
        rouge_std[method] = np.std(rouge_scores[method], axis=0)
    return rouge_mean, rouge_median, rouge_std
