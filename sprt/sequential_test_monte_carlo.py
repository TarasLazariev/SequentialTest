import numpy as np
from tqdm.notebook import tqdm

from sprt.ab_stat_tests import calc_prop_test_sample_size, calc_prop_test_pvalue
from sprt.sequential_test import Sequential_test

def sequential_test_simulation(base_cr, 
                               mde, 
                               mde_real, 
                               alpha=0.1, 
                               power=0.9, 
                               n_simulations=10000, 
                               simulation_step=100):
    """
    Execute n_simulations to estimate the real alpha, beta of the test and the expected sample size
    """
    cr1 = base_cr
    cr2 = cr1 * (1 + mde)
    cr2_real = cr1 * (1 + mde_real)
    beta = 1 - power
    data_a = np.random.choice([0,1], 100000, p=[1 - cr1, cr1])
    data_b = np.random.choice([0,1], 100000, p=[1 - cr2_real, cr2_real])
    sample_size = int(calc_prop_test_sample_size(cr=cr1, mde=mde, power=power, alpha=alpha, alternative='greater'))

    seq_result_aa, seq_result_ab = [], []
    sample_size_ab, sample_size_aa = [], []
    times_classic_aa, times_classic_ab = 0, 0

    for it in tqdm(range(n_simulations)):
        a1 = np.random.choice(data_a, size=sample_size, replace=False)
        a2 = np.random.choice(data_a, size=sample_size, replace=False)
        b = np.random.choice(data_b, size=sample_size, replace=False)

        for i in range(500, sample_size, simulation_step):
            a1_temp = a1[:i]
            a2_temp = a2[:i]
            test_aa = Sequential_test(a1_temp, a2_temp, alpha=alpha, beta=beta).calculate_
            if test_aa==1:
                seq_result_aa.append(1)
                sample_size_aa.append(i)
                break
            elif test_aa==0:
                seq_result_aa.append(0)
                sample_size_aa.append(i)
                break
        if len(seq_result_aa) < it + 1:
            times_classic_aa += 1
            if calc_prop_test_pvalue(sum(a2), sample_size, sum(a1), sample_size) < alpha:
                seq_result_aa.append(1)
            else:
                seq_result_aa.append(0)
                sample_size_aa.append(sample_size)

        for i in range(500, sample_size, simulation_step):
            a1_temp = a1[:i]
            b_temp = b[:i]
            test_ab = Sequential_test(a1_temp, b_temp, alpha=alpha, beta=beta).calculate_
            if test_ab==1:
                seq_result_ab.append(1)
                sample_size_ab.append(i)
                break
            elif test_ab==0:
                seq_result_ab.append(0)
                sample_size_ab.append(i)
                break
        if len(seq_result_ab) < it + 1:
            times_classic_ab += 1
            if calc_prop_test_pvalue(sum(b), sample_size, sum(a1), sample_size) < alpha:
                seq_result_ab.append(1)
                sample_size_ab.append(sample_size)
            else:
                seq_result_ab.append(0)
                
    print('Type I Error', round(np.sum(seq_result_aa) / n_simulations, 2))
    print('Type II Error', round((len(seq_result_ab) - np.sum(seq_result_ab)) / n_simulations, 2))
    print('-----------------------------------')
    print('Stopped earlier if H0 True', round(1 - times_classic_aa / n_simulations, 2))
    print('Stopped earlier if H1 True', round(1 - times_classic_ab / n_simulations,2))
    print('-----------------------------------')
    print('Saved time if H0 True', round((sample_size - np.mean(sample_size_aa)) / sample_size, 2))
    print('Saved time if H1 True', round((sample_size - np.mean(sample_size_ab)) / sample_size, 2))