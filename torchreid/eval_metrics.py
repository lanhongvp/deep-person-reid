from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import copy
from collections import defaultdict
import sys
import pandas as pd

try:
    from torchreid.eval_lib.cython_eval import eval_market1501_wrap
    CYTHON_EVAL_AVAI = True
    print("Cython evaluation is AVAILABLE")
except ImportError:
    CYTHON_EVAL_AVAI = False
    print("Warning: Cython evaluation is UNAVAILABLE")


def eval_cuhk03(distmat, q_vids, g_vids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query vid and camid
        q_vid = q_vids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same vid and camid with query
        order = indices[q_idx]
        remove = (g_vids[order] == q_vid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_vids = g_vids[order][keep]
        g_vids_dict = defaultdict(list)
        for idx, vid in enumerate(kept_g_vids):
            g_vids_dict[vid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_vids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_vids, g_vids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query vid and camid
        q_vid = q_vids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same vid and camid with query
        order = indices[q_idx]
        remove = (g_vids[order] == q_vid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def evaluate_aicity(distmat,q_ids,g_ids,max_rank=100,exp='exp0',vis_ranked_res=False):
    q_num = distmat.shape[0]
    g_num = distmat.shape[1]
    aicity_results = {}

    for idx,q_id in enumerate(q_ids):
        distmat_cmp = np.argsort(distmat[idx,:])
        distmat_rank = distmat_cmp[:max_rank]
        distmat_rank = g_ids[distmat_rank]
        aicity_results[q_id] = distmat_rank
    
    aicity_results = pd.DataFrame(list(aicity_results.items()),columns=['query_ids','gallery_ids'])
    # embed()
    aicity_results = aicity_results.sort_values('query_ids')
    _write2txt(aicity_results,exp)
    if vis_ranked_res:
        arr_aicity_results = np.array(aicity_reuslts)



def _write2txt(aicity_results,exp):
    with open(exp+'.txt','w') as f:
        for idx in range(len(aicity_results)):
            sep_c = ' '
            row_ranks = []
            idx_row = aicity_results.iloc[idx]['gallery_ids'][:100]
            # embed()
            for item in idx_row:
                row_rank = str(item)
                row_ranks.append(row_rank)
            sep_c = sep_c.join(row_ranks)
            # embed()
            sep_c = sep_c+'\n'
            # embed()
            f.write(sep_c)
        f.close()


def evaluate(distmat, q_vids, g_vids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False, use_cython=True):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_vids, g_vids, q_camids, g_camids, max_rank)
    else:
        if use_cython and CYTHON_EVAL_AVAI:
            return eval_market1501_wrap(distmat, q_vids, g_vids, q_camids, g_camids, max_rank)
        else:
            return eval_market1501(distmat, q_vids, g_vids, q_camids, g_camids, max_rank)
