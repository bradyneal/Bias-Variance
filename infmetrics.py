from itertools import combinations
from scipy.spatial.distance import hamming


def get_pairwise_hamming_dists(seqs):
    """
    Return all n choose 2 pairwise hamming distances in the first return value
    and return, in the second return value, a list of n length n - 1 lists,
    where the ijth element in this list of lists is the hamming distance
    between the the ith and jth sequence.
    """
    num_seqs = len(seqs)
    hamm_dist_lists = [[] for _ in xrange(num_seqs)]
    all_hamm_dists = []
    for i in xrange(num_seqs):
        for j in xrange(i + 1, num_seqs):
            hamm_dist = hamming(seqs[i], seqs[j])
            hamm_dist_lists[i].append(hamm_dist)
            hamm_dist_lists[j].append(hamm_dist)
            all_hamm_dists.append(hamm_dist)
    return all_hamm_dists, hamm_dist_lists
