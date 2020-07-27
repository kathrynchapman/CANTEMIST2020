import numpy as np
from itertools import combinations
from math import e


def plackett_luce(some_list):

    for i in range(1,len(some_list)):
        some_list[i] /= np.sum(some_list[i:])
    return np.sum(np.log(some_list))





if __name__=='__main__':

    pred_probs = np.random.rand(8,495)

    true_ranks = [0]*495
    true_ranks[100] = 4
    true_ranks[400] = 3
    true_ranks[300] = 2
    true_ranks[56] = 1

    n_labels = max(true_ranks)
    true_label_probs = np.flip(pred_probs[np.argsort(true_ranks)[-n_labels:]]) # these are the first n elements of our new array
    # now, we want to concatenate this with the remaining elements, which are sorted in descending order

    remaining_probs = np.flip(np.sort(pred_probs[np.argsort(true_ranks)[:-n_labels]]))

    new_array = np.concatenate((true_label_probs, remaining_probs))

    print(plackett_luce(new_array)) # THIS!!!!! We want to minimize this!!
