import numpy as np

def calculate_oscr(gt, scores, unk_label=-1):
    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes] or [N_samples, N_classes+1]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = gt.astype(int)
    kn = gt >= 0
    unk = gt == unk_label

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr = [], []
    pred_class = np.argmax(scores, axis=1)
    max_score = np.max(scores, axis=1)
    target_score = scores[kn][range(kn.sum()), gt[kn]]
    print(kn.sum())
    print(target_score)
    print("######")
    print(np.unique(target_score)[:-1])

    for tau in np.unique(target_score)[:-1]:
        val = ((pred_class[kn] == gt[kn]) & (target_score > tau)).sum() / total_kn
        ccr.append(val)

        val = (unk & (max_score > tau)).sum() / total_unk
        print(val)
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    return ccr, fpr

rng = np.random.default_rng(seed=43)
scores = rng.random((9,3))

print(scores.shape)
gt = np.array([1, 0, 0 ,1, 1, 0, -1, 0, -1])

print(calculate_oscr(gt, scores))
