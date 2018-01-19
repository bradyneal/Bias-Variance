from __future__ import division
from itertools import combinations
from scipy.spatial.distance import hamming
import numpy as np
import torch
from functools import partial


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


def get_pairwise_agreements(seqs):
    """"Same as above, but using the 'agreement' metric defined below"""
    return get_pairwise_dists(seqs, get_agreement)


def get_pairwise_disagreements(seqs):
    """"Same as above, but using the 'disagreement' metric defined below"""
    return get_pairwise_dists(seqs, get_disagreement)


def get_pairwise_weight_dists(seqs):
    """"Same as above, but using the distance in weight space"""
    return get_pairwise_dists(seqs, get_weight_dist)


def get_agreement(seq1, seq2, mis_label=0):
    """
    Return the symmetric 'agreement' between two sequences where
    mis_label is the label that indicates an example was misclassified.
    """
    mis1 = set(np.nonzero(seq1 == mis_label)[:, 0])
    mis2 = set(np.nonzero(seq2 == mis_label)[:, 0])
    num_mis_both = len(mis1.intersection(mis2))
    agree1 = num_mis_both / len(mis1)
    agree2 = num_mis_both / len(mis2)
    sym_agreement = (agree1 + agree2) / 2
    return sym_agreement


def get_disagreement(seq1, seq2, mis_label=0):
    """
    Return the symmetric 'disagreement' between two sequences where
    mis_label is the label that indicates an example was misclassified.
    """
    return 1 - get_agreement(seq1, seq2, mis_label=0)


def get_weight_dist(w1, w2, p=2):
    """Get distance between models in weight space"""
    return torch.norm(w1 - w2, p=p)


def expected_hamming(p):
    """
    Return the expected hamming distance between two random vectors X and Y,
    under the following assumptions:
    1. P(X_i = Y_i) = P(X_j = Y_j). In words, this means (X_i, Y_i) and
    (X_j, Y_j) are identically jointly distributed. In other words, all data
    points are equally easy (or hard) to learn (this is an empirically false
    assumption).
    2. X_i and Y_i are conditionally independent (conditioned on i). In other
    words, the predictions between any two learned models on the same test
    example are independent (obviously false assumption).
    """
    return 2 * p * (1 - p)


def hamming_diff(seq1, seq2, p):
    """
    Return the difference between the hamming distance of the two sequences and
    the expected hamming distance between two random vectors.
    """
    return hamming(seq1, seq2) - expected_hamming(p)
