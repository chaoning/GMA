# -*- coding:utf8 -*-

import numpy as np
from scipy import linalg
from pysnptools.snpreader import Bed
import pandas as pd
import datetime
from scipy.stats import chi2


from cal_kin_val import *


def gma_univariate_eigen_lt_gwas(y, xmat, bed_file, out_file=None, init=None, maxiter=100, cc=1.0e-8):
    
    # kinship
    print 'Build the kinship matrix'
    starttime = datetime.datetime.now()
    num_id = max(y.shape)
    snp_on_disk = Bed(bed_file, count_A1=False)
    snp_mat = snp_on_disk.read().val
    freq = np.sum(snp_mat, axis=0) / (2 * snp_on_disk.iid_count)
    freq.shape = (1, snp_on_disk.sid_count)
    snp_mat = snp_mat - 2 * freq
    scale = 2 * freq * (1 - freq)
    scale = np.sum(scale)
    kin = np.dot(snp_mat, snp_mat.T) / scale
    
    endtime = datetime.datetime.now()
    print "Running time", (endtime - starttime).seconds
    print 'Finish'
    
    print 'Eigen decomposition'
    starttime = datetime.datetime.now()
    kin_eigen_val, kin_eigen_vec = linalg.eigh(kin)
    kin_eigen_val = kin_eigen_val.reshape(len(kin_eigen_val), 1)
    endtime = datetime.datetime.now()
    print "Running time", (endtime - starttime).seconds
    print 'Finish'
    y = np.dot(kin_eigen_vec.T, y)
    xmat = np.dot(kin_eigen_vec.T, xmat)
    if init is not None:
        var = np.array(init)
    else:
        var = np.array([1.0, 1.0])
    
    fd_mat = np.zeros(2)
    ai_mat = np.zeros((2, 2))
    em_mat = np.zeros((2, 2))
    ### 计算null model的方差组分
    
    print 'Estimate variances'
    starttime = datetime.datetime.now()
    for i in range(maxiter):
        print 'Start the iteration:', i+1
        vmat = 1.0 / (kin_eigen_val * var[0] + var[1])
        vx = np.multiply(vmat, xmat)
        xvx = np.dot(xmat.T, vx)
        xvx = np.linalg.inv(xvx)
        
        # py
        xvy = np.dot(vx.T, y)
        y_xb = y - np.dot(xmat, np.dot(xvx, xvy))
        py = np.multiply(vmat, y_xb)
        
        # add_py p_add_py
        add_py = np.multiply(kin_eigen_val, py)
        xvy = np.dot(vx.T, add_py)
        y_xb = add_py - np.dot(xmat, np.dot(xvx, xvy))
        p_add_py = np.multiply(vmat, y_xb)
        
        # res_py p_res_py
        res_py = py.copy()
        xvy = np.dot(vx.T, res_py)
        y_xb = res_py - np.dot(xmat, np.dot(xvx, xvy))
        p_res_py = np.multiply(vmat, y_xb)
        
        # fd
        tr_vd = np.sum(np.multiply(vmat, kin_eigen_val))
        xvdvx = np.dot(xmat.T, vmat*kin_eigen_val*vx)
        tr_2d = np.sum(np.multiply(xvdvx, xvx))
        ypvpy = np.sum(np.dot(py.T, add_py))
        fd_mat[0] = 0.5*(-tr_vd + tr_2d + ypvpy)
        
        tr_vd = np.sum(vmat)
        xvdvx = np.dot(xmat.T, vmat * vx)
        tr_2d = np.sum(np.multiply(xvdvx, xvx))
        ypvpy = np.sum(np.dot(py.T, res_py))
        fd_mat[1] = 0.5 * (-tr_vd + tr_2d + ypvpy)
        
        # AI
        ai_mat[0, 0] = np.sum(np.dot(add_py.T, p_add_py))
        ai_mat[0, 1] = ai_mat[1, 0] = np.sum(np.dot(add_py.T, p_res_py))
        ai_mat[1, 1] = np.sum(np.dot(res_py.T, p_res_py))
        ai_mat = 0.5*ai_mat
        
        # EM
        em_mat[0, 0] = num_id / (var[0] * var[0])
        em_mat[1, 1] = num_id / (var[1] * var[1])
        
        print "FD:", fd_mat
        print "AI:", ai_mat
        print "EM:", em_mat
        
        for j in range(0, 51):
            gamma = j * 0.02
            wemai_mat = (1 - gamma)*ai_mat + gamma*em_mat
            delta = np.dot(linalg.inv(wemai_mat), fd_mat)
            var_update = var + delta
            if min(var_update) > 0:
                print 'EM weight value:', gamma
                break
        
        print 'Updated variances:', var_update
        
        # Convergence criteria
        cc_val = np.sum(pow(delta, 2)) / np.sum(pow(var_update, 2))
        cc_val = np.sqrt(cc_val)
        var = var_update.copy()
        
        print "CC: ", cc_val
        if cc_val < cc:
            break
    
    endtime = datetime.datetime.now()
    print "Running time", (endtime - starttime).seconds
    print 'Finish'
    
    # GWAS
    
    print 'Start GWAS'
    starttime = datetime.datetime.now()
    vmat = 1.0 / (kin_eigen_val * var[0] + var[1])
    vx = np.multiply(vmat, xmat)
    xvx = np.dot(xmat.T, vx)
    xvx = np.linalg.inv(xvx)
    
    # py
    xvy = np.dot(vx.T, y)
    y_xb = y - np.dot(xmat, np.dot(xvx, xvy))
    py = np.multiply(vmat, y_xb)
    
    snp_mat = np.dot(kin_eigen_vec.T, snp_mat)
    # 效应
    chi_vec = []
    p_vec = []
    eff_vec = np.dot(snp_mat.T, py)*var[0]
    eff_vec = eff_vec[:, -1]
    for i in range(snp_on_disk.sid_count):
        snpi = snp_mat[:, i:(i+1)]
        snp_var1 = np.sum(reduce(np.multiply, [snpi, vmat, snpi]))
        snp_var2 = np.dot(snpi.T, vx)
        snp_var2 = reduce(np.dot, [snp_var2, xvx, snp_var2.T])
        snp_var = (snp_var1 + np.sum(snp_var2)) * var[0] * var[0]
        chi_val = eff_vec[i]*eff_vec[i]/snp_var
        p_val = chi2.sf(chi_val, 1)
        chi_vec.append(chi_val)
        p_vec.append(p_val)
    
    endtime = datetime.datetime.now()
    print "Running time", (endtime - starttime).seconds
    print 'Finish'
    
    snp_info_file = bed_file + '.bim'
    snp_info = pd.read_csv(snp_info_file, sep='\s+', header=None)
    res_df = snp_info.iloc[:, [0, 1, 3, 4, 5]]
    res_df.columns = ['chro', 'snp_ID', 'pos', 'allele1', 'allele2']
    res_df.loc[:, 'eff_val'] = eff_vec
    res_df.loc[:, 'chi_val'] = chi_vec
    res_df.loc[:, 'p_val'] = p_vec
    
    if out_file is not None:
        try:
            res_df.to_csv(out_file, sep=' ', index=False)
        except Exception, e:
            print e
            print 'Fail to output the result!'
            exit()
    
    return res_df
