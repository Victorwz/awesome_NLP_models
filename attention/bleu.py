import math
import sys

from collections import Counter
from functools import reduce


def compute_bleu(reflists, hyps, n_max=4, use_shortest_ref=False):
    assert len(reflists) == len(hyps)

    prec_mean = 1  # TODO: Implement
    brevity_penalty = 0  # TODO:Implement

    N = len(hyps)
    for i in range(1, n_max+1):
        a_sum = b_sum = 0
        for j in range(N):
            a_sum += get_ngram_counts(reflists[j], hyps[j], i)[0]
            b_sum += get_ngram_counts(reflists[j], hyps[j], i)[1]
        prec_mean *= (a_sum / b_sum)

    H = R = 0
    
    for i, refs in enumerate(reflists):
        H += len(hyps[i])
        ref_len = map(lambda ref: len(ref), refs)
        R += min(ref_len, key = lambda reflen: abs(reflen - len(hyps[i])))

    prec_mean = pow(prec_mean, 1/4)
    brevity_penalty = min(1, math.exp(1 - (R / H)))

    bleu = brevity_penalty * prec_mean

    return bleu


def get_ngram_counts(refs, hyp, n):
    hyp_ngrams = [tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)]
    num_hyp_ngrams = max(1, len(hyp_ngrams))  # Avoid empty
    M = len(refs)

    num_hyp_ngrams_in_refs_clipped = 0

    num_hyp_ngrams_in_h = {}

    for ngram in hyp_ngrams:
        try:
            num_hyp_ngrams_in_h[ngram][0] += 1
        except Exception:
            num_hyp_ngrams_in_h[ngram] = [1, 0]

    for ref in refs:
        ref_ngrams = [tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)]
        num_hyp_ngrams_in_ref = Counter()
        for ngram in ref_ngrams:
            try:
                g_rj = num_hyp_ngrams_in_h[ngram][1]
                num_hyp_ngrams_in_ref[ngram] += 1
                num_hyp_ngrams_in_h[ngram][1] = max(g_rj, num_hyp_ngrams_in_ref[ngram])
            except Exception:
                pass

    for ngram, count in num_hyp_ngrams_in_h.items():
        num_hyp_ngrams_in_refs_clipped += min(count)

    return num_hyp_ngrams_in_refs_clipped, num_hyp_ngrams
