from typing import Dict
from collections import defaultdict

import numpy as np 
from Bio import pairwise2

def align_chains(
    src_chain_to_seq: Dict[str, str],
    tgt_chain_to_seq: Dict[str, str]
):
    chain_alns = defaultdict(dict)
    for sc, sseq in src_chain_to_seq.items():
        for tc, tseq in tgt_chain_to_seq.items():
            aln = global_align(sseq, tseq)
            if aln is not None and aln["aln"].score > 0:
                chain_alns[sc][tc] = aln
    return chain_alns

def global_align(
    srcseq: str,
    tgtseq: str,
    match_reward: float = 2,
    mismatch_penalty: float = -1,
    gap_penalty: float = -2.,
    affine_penalty: float = -0.5,                   
):
    def _get_alnidx(seqA, seqB):
        srcidx, tgtidx = [], []
        ai, bj = 0, 0
        for a, b in zip(seqA, seqB):
            if a != '-' and b != '-':
                srcidx.append(ai)
                tgtidx.append(bj)
            if a != '-':
                ai += 1
            if b != '-':
                bj += 1
        srcidx = np.array(srcidx, dtype=np.int64)
        tgtidx = np.array(tgtidx, dtype=np.int64)
        return srcidx, tgtidx

    srcseq = ''.join([_ for _ in srcseq if _ != '-'])
    tgtseq = ''.join([_ for _ in tgtseq if _ != '-'])

    aln = pairwise2.align.globalms(
        srcseq,
        tgtseq,
        match_reward,
        mismatch_penalty,
        gap_penalty,
        affine_penalty
    )
    if len(aln) >= 1:
        aln = aln[0]
        srcidx, tgtidx = _get_alnidx(aln.seqA, aln.seqB)
        return {
            'aln': aln, 
            'src': srcidx, 
            'tgt': tgtidx, 
        } 
    else:
        return