# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import six
import itertools
import numpy as np
import spacy
import utils

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


def _len_lcs(x, y):
    """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def multi_rouge_n(sequences, scores_ids, n=2):
    """
    Efficient way to compute highly repetitive scoring
    i.e. sequences are involved multiple time
    Args:
        sequences(list[str]): list of sequences (either hyp or ref)
        scores_ids(list[tuple(int)]): list of pairs (hyp_id, ref_id)
            ie. scores[i] = rouge_n(scores_ids[i][0],
                                    scores_ids[i][1])
    Returns:
        scores: list of length `len(scores_ids)` containing rouge `n`
                scores as a dict with 'f', 'r', 'p'
    Raises:
        KeyError: if there's a value of i in scores_ids that is not in
                  [0, len(sequences)[
    """
    ngrams = [_get_word_ngrams(n, sequence) for sequence in sequences]
    counts = [len(ngram) for ngram in ngrams]

    scores = []
    for hyp_id, ref_id in scores_ids:
        evaluated_ngrams = ngrams[hyp_id]
        evaluated_count = counts[hyp_id]

        reference_ngrams = ngrams[ref_id]
        reference_count = counts[ref_id]

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        scores += [f_r_p_rouge_n(evaluated_count,
                                 reference_count, overlapping_count)]
    return scores


def rouge_n(evaluated_sentences, reference_sentences, n=2, embed=False, embed_dict=None):
    """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.
    Returns:
      A tuple (f1, precision, recall) for ROUGE-N
    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    if embed:
        return f_r_p_rouge_n_embed(evaluated_count, reference_count, evaluated_ngrams, reference_ngrams, embed_dict)
    else:
        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        return f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count)


def f_r_p_rouge_n_embed(evaluated_count, reference_count, evaluated_ngrams, reference_ngrams, embed_dict=None):
    if embed_dict:
        nlp = spacy.load(embed_dict)
    else:
        nlp = spacy.load('en_core_web_lg')

    # get word vector matrix for all evaluated ngrams
    eval_wv_matrix = []
    for eval in evaluated_ngrams:
        eval_wv = None
        for token in eval:
            eval_token = nlp(token)
            if eval_token.has_vector:
                if (eval_wv is None) and (eval_token.vector_norm > 0):
                    eval_wv = eval_token.vector / eval_token.vector_norm
                else:
                    if eval_token.vector_norm > 0:
                        eval_wv += eval_token.vector / eval_token.vector_norm
                        eval_wv /= np.linalg.norm(eval_wv)
        if eval_wv is not None: eval_wv_matrix.append(eval_wv)

    # get word vector matrix for all reference ngrams
    ref_wv_matrix = []
    for ref in reference_ngrams:
        ref_wv = None
        for token in ref:
            ref_token = nlp(token)
            if ref_token.has_vector:
                if (ref_wv is None) and (ref_token.vector_norm > 0): # to avoid 0 division
                    ref_wv = ref_token.vector / ref_token.vector_norm
                else:
                    if ref_token.vector_norm > 0:
                        ref_wv += ref_token.vector / ref_token.vector_norm
                        ref_wv /= np.linalg.norm(ref_wv)
        if ref_wv is not None: ref_wv_matrix.append(ref_wv)

    # calculate maximum of the similarity score
    similarity_matrix = np.dot(np.array(eval_wv_matrix), np.array(ref_wv_matrix).T)
    eval_score = np.sum(np.max(similarity_matrix, axis=1))
    ref_score = np.sum(np.max(similarity_matrix, axis=0))

    # calculate metrics
    precision = eval_score / evaluated_count
    recall = ref_score / reference_count
    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return {"f": f1_score, "p": precision, "r": recall}


def f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count):
    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return {"f": f1_score, "p": precision, "r": recall}


def _union_lcs(evaluated_sentences, reference_sentence, prev_union=None, embed=False, embed_dict=None):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C.
    For example:
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8
    and c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1
    is "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5".
    The union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5"
    and LCS_u(r_i, C) = 4/5.
    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries
    Returns:
      float: LCS_u(r_i, C)
    ValueError:
      Raises exception if a param has len <= 0
    """
    if prev_union is None:
        prev_union = set()

    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = prev_union
    prev_count = len(prev_union)
    reference_words = _split_into_words([reference_sentence])

    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    new_lcs_count = len(lcs_union) - prev_count
    return new_lcs_count, lcs_union


def _union_lcs_embed(evaluated_sentences, reference_sentences, embed_dict):
    if embed_dict:
        nlp = spacy.load(embed_dict)
    else:
        nlp = spacy.load('en_core_web_lg')

    reference_words = set(_split_into_words(reference_sentences))
    evaluated_words = set(_split_into_words(evaluated_sentences))

    # get word vector matrix for all evaluated words
    eval_wv_matrix = []
    for eval in evaluated_words:
        eval_token = nlp(eval)
        if (eval_token.has_vector) and (eval_token.vector_norm > 0):
            eval_wv_matrix.append(eval_token.vector / eval_token.vector_norm)

    # get word vector matrix for all referene words
    ref_wv_matrix = []
    for ref in reference_words:
        ref_token = nlp(ref)
        if (ref_token.has_vector) and (ref_token.vector_norm > 0):
            ref_wv_matrix.append(ref_token.vector / ref_token.vector_norm)

    # calculate maximum of the similarity score
    similarity_matrix = np.dot(np.array(eval_wv_matrix), np.array(ref_wv_matrix).T)
    lcs_count = np.sum(np.max(similarity_matrix, axis=0))
    return lcs_count


def rouge_l_summary_level(evaluated_sentences, reference_sentences, embed=False, embed_dict=None):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary
    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries
    Returns:
      A float: F_lcs
    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # total number of words in reference sentences
    m = len(set(_split_into_words(reference_sentences)))

    # total number of words in evaluated sentences
    n = len(set(_split_into_words(evaluated_sentences)))

    # print("m,n %d %d" % (m, n))
    union_lcs_sum_across_all_references = 0
    union = set()
    if embed:
        llcs = _union_lcs_embed(evaluated_sentences, reference_sentences, embed_dict)
    else:
        for ref_s in reference_sentences:
            lcs_count, union = _union_lcs(evaluated_sentences, ref_s, prev_union=union)
            union_lcs_sum_across_all_references += lcs_count
        llcs = union_lcs_sum_across_all_references

    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return {"f": f_lcs, "p": p_lcs, "r": r_lcs}

class Rouge:
    DEFAULT_METRICS = ["rouge-1", "rouge-2", "rouge-l"]
    AVAILABLE_METRICS = {
        "rouge-1": lambda hyp, ref, rouge_embed, embed_dict: rouge_n(hyp, ref, 1, rouge_embed, embed_dict),
        "rouge-2": lambda hyp, ref, rouge_embed, embed_dict: rouge_n(hyp, ref, 2, rouge_embed, embed_dict),
        "rouge-l": lambda hyp, ref, rouge_embed, embed_dict: rouge_l_summary_level(hyp, ref, rouge_embed, embed_dict),
    }
    DEFAULT_STATS = ["f", "p", "r"]
    AVAILABLE_STATS = ["f", "p", "r"]

    def __init__(self, metrics=None, stats=None):
        if metrics is not None:
            self.metrics = [m.lower() for m in metrics]

            for m in self.metrics:
                if m not in Rouge.AVAILABLE_METRICS:
                    raise ValueError("Unknown metric '%s'" % m)
        else:
            self.metrics = Rouge.DEFAULT_METRICS

        if stats is not None:
            self.stats = [s.lower() for s in stats]

            for s in self.stats:
                if s not in Rouge.AVAILABLE_STATS:
                    raise ValueError("Unknown stat '%s'" % s)
        else:
            self.stats = Rouge.DEFAULT_STATS

    def get_scores(self, hyps, refs, avg=False, ignore_empty=False, rouge_embed=False, embed_dict=None):
        if isinstance(hyps, six.string_types):
            hyps, refs = [hyps], [refs]

        if ignore_empty:
            # Filter out hyps of 0 length
            hyps_and_refs = zip(hyps, refs)
            hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
            hyps, refs = zip(*hyps_and_refs)

        assert(type(hyps) == type(refs))
        assert(len(hyps) == len(refs))

        if not avg:
            return self._get_scores(hyps, refs, rouge_embed, embed_dict)
        return self._get_avg_scores(hyps, refs, rouge_embed, embed_dict)

    def _get_scores(self, hyps, refs, rouge_embed, embed_dict):
        scores = []
        for hyp, ref in zip(hyps, refs):
            sen_score = {}
            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, rouge_embed, embed_dict)
                sen_score[m] = {s: sc[s] for s in self.stats}
            scores.append(sen_score)
        return scores

    def _get_avg_scores(self, hyps, refs, rouge_embed, embed_dict):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}

        count = 0
        for (hyp, ref) in zip(hyps, refs):
            try:
                hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
                ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
            except:
                pass

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, rouge_embed, embed_dict)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}
            count += 1
        scores = {m: {s: scores[m][s] / count for s in scores[m]}
                  for m in scores}
        return scores
