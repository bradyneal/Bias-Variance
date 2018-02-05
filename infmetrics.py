from __future__ import print_function, division
from itertools import combinations
from scipy.spatial.distance import hamming
import numpy as np
import torch
from functools import partial
from math import sqrt

MISCLASS_LABEL = 0
CORRECT_CLASS_LABEL = 1 - MISCLASS_LABEL


def get_pairwise_dists(seqs, metric):
    """
    seqs: list of sequences to measure pairwise distances between
    metric: specific distance metric used between all pairs of seqs
    
    Return all n choose 2 pairwise distances in the first return value
    and return, in the second return value, a list of n length n - 1 lists,
    where the ijth element in this list of lists is the distance
    between the the ith and jth sequence
    """
    num_seqs = len(seqs)
    dist_lists = [[] for _ in xrange(num_seqs)]
    all_dists = []
    for i in xrange(num_seqs):
        for j in xrange(i + 1, num_seqs):
            dist = metric(seqs[i], seqs[j])
            dist_lists[i].append(dist)
            dist_lists[j].append(dist)
            all_dists.append(dist)
    return all_dists, dist_lists


def get_pairwise_hamming_dists(seqs):
    """
    Return all n choose 2 pairwise hamming distances in the first return value
    and return, in the second return value, a list of n length n - 1 lists,
    where the ijth element in this list of lists is the hamming distance
    between the the ith and jth sequence.
    """
    return get_pairwise_dists(seqs, hamming)


def get_pairwise_hamming_diffs(seqs, p):
    """
    Same as above, but using the relative hamming distance metric defined below
    """
    return get_pairwise_dists(seqs, partial(hamming_diff, p=p))


def get_pairwise_pos_agreements(seqs):
    """"Same as above, but using the 'positive agreement' metric defined below"""
    return get_pairwise_dists(seqs, get_pos_agreement)


def get_pairwise_neg_agreements(seqs):
    """"Same as above, but using the 'negative agreement' metric defined below"""
    return get_pairwise_dists(seqs, get_neg_agreement)


def get_pairwise_pos_disagreements(seqs):
    """"Same as above, but using the 'positive disagreement' metric defined below"""
    return get_pairwise_dists(seqs, get_pos_disagreement)


def get_pairwise_neg_disagreements(seqs):
    """"Same as above, but using the 'negative disagreement' metric defined below"""
    return get_pairwise_dists(seqs, get_neg_disagreement)


def get_pairwise_disagreements(seqs):
    """"Same as above, but using the 'disagreement' metric defined below"""
    return get_pairwise_dists(seqs, get_disagreement)


def get_pairwise_weight_dists(seqs):
    """"Same as above, but using the distance in weight space"""
    return get_pairwise_dists(seqs, get_weight_dist)


def get_pairwise_weight_dists_normalized(seqs):
    """"Same as above, but using the normalized distance in weight space"""
    return get_pairwise_dists(seqs, get_weight_dist_normalized)


def get_agreement(seq1, seq2, label):
    """
    Return the symmetric 'agreement' between two sequences where
    we are conditioning only on the labels with 'label'.
    """
    cond1 = set(np.nonzero(seq1 == label)[:, 0])
    cond2 = set(np.nonzero(seq2 == label)[:, 0])
    num_cond_both = len(cond1.intersection(cond2))
    agree1 = num_cond_both / len(cond1)
    agree2 = num_cond_both / len(cond2)
    sym_agreement = (agree1 + agree2) / 2
    return sym_agreement


def get_pos_agreement(seq1, seq2, cor_label=CORRECT_CLASS_LABEL):
    """
    Return the symmetric 'agreement' between two sequences where
    cor_label is the label that indicates an example was correctly classified.
    """
    return get_agreement(seq1, seq2, label=cor_label)


def get_neg_agreement(seq1, seq2, mis_label=MISCLASS_LABEL):
    """
    Return the symmetric 'agreement' between two sequences where
    mis_label is the label that indicates an example was misclassified.
    """
    return get_agreement(seq1, seq2, label=mis_label)


def get_pos_disagreement(seq1, seq2, cor_label=CORRECT_CLASS_LABEL):
    """
    Complement of positive agreement above.
    Note: this is the same as if the symmetric part were to be done between
    disagreements, rather than between agreements.
    """
    return 1 - get_pos_agreement(seq1, seq2, cor_label=cor_label)


def get_neg_disagreement(seq1, seq2, mis_label=MISCLASS_LABEL):
    """
    Complement of negative agreement above.
    Note: this is the same as if the symmetric part were to be done between
    disagreements, rather than between agreements.
    """
    return 1 - get_neg_agreement(seq1, seq2, mis_label=mis_label)


def get_weight_dist(w1, w2, p=2):
    """Return the distance between models in weight space"""
    return torch.norm(w1 - w2, p=p)


def get_weight_dist_normalized(w1, w2, p=2):
    """
    Return the distance between models in weight space, normalized by the
    square root of the dimension
    """
    return get_weight_dist(w1, w2, p=p) / sqrt(len(w1))


def expected_hamming(p1, p2=None):
    """
    Return the expected hamming distance between two random vectors X and Y
    where X_i ~ Bernoulli(p1) and Y_i ~ Bernoulli(p2) (defaults to p1 if p2
    isn't specified), under the following assumptions:
    1. P(X_i = Y_i) = P(X_j = Y_j). In words, this means (X_i, Y_i) and
    (X_j, Y_j) are identically jointly distributed. In other words, all data
    points are equally easy (or hard) to learn (this is an empirically false
    assumption).
    2. X_i and Y_i are conditionally independent (conditioned on i). In other
    words, the predictions between any two learned models on the same test
    example are independent (obviously false assumption).
    """
    if p2 is None:
        return 2 * p1 * (1 - p1)
    else:
        return p1 + p2 - 2 * p1 * p2


def hamming_diff(seq1, seq2, p1=None, p2=None):
    """
    Return the difference between the hamming distance of the two sequences and
    the expected hamming distance between two random vectors.
    If p is specified, use it as the common p for both vector. Otherwise,
    calculate each vector's respective p and use those.
    """
    if p1 is not None and p2 is not None:
        expected = expected_hamming(p1, p2)
    elif p1 is not None:
        expected = expected_hamming(p1)
    elif p1 is None and p2 is None:
        p1 = torch.mean(seq1)
        p2 = torch.mean(seq2)
        expected = expected_hamming(p1, p2)
    else:
        raise ValueError('Invalid arguments: p1 is None and p2 is not None')
    return hamming(seq1, seq2) - expected
    

def hamming_std(n, p1, p2=None):
    """
    Return the standard deviation under the following assumptions:
    Same 1 and 2 as for the expectation.
    3. P(X_i = Y_i) and P(X_j = Y_j) are independent. Given assumptions 1 and 2,
    this amounts to adding the assumption that the ith and jth predictions from
    the same classifier are the independent (obviously false assumption).
    """
    if p2 is None:
        p = p1
        return sqrt((2*p - 6*p**2 + 8*p**3 - 4*p**4) / n)
    else:
        numer = p1 + p2 - 4*p1*p2 - p1**2 - p2**2 + 4*(p2*p1**2 + p1*p2**2) - 4*p1**2 * p2**2
        return sqrt(numer / n)
