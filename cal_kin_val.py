import numpy as np
from pysnptools.snpreader import Bed
import pandas as pd

def cal_kin_val(bed_file, small_val=0.001):
    snp_on_disk = Bed(bed_file, count_A1=False)
    snp_mat = snp_on_disk.read().val
    freq = np.sum(snp_mat, axis=0) / (2 * snp_on_disk.iid_count)
    freq.shape = (1, snp_on_disk.sid_count)
    snp_mat = snp_mat - 2*freq
    scale = 2 * freq * (1 - freq)
    scale = np.sum(scale)
    kin = np.dot(snp_mat,snp_mat.T)/scale
    kin_diag = np.diag(kin)
    kin_diag = kin_diag + kin_diag * small_val
    np.fill_diagonal(kin, kin_diag)
    return kin
