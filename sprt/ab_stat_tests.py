from scipy.stats import norm
from statsmodels.stats.proportion import test_proportions_2indep
import numpy as np

def calc_prop_test_sample_size(cr, mde, alpha=0.05, power=0.8, alternative='greater'):
    """calculates sample size for 1 variation for a proportional test
    :param cr: historical conversion rate
    :param mde: expected mde
    :param alpha: alpha test parameter
    :param power: power test parameter
    :param alternative: hypotesis type - one-sided, less, greater, two-sided
    :return: sample size
    """
    if alternative=='two-sided':
        alpa_coef = 1 - alpha / 2
    elif alternative in ('greater', 'less', 'one-sided'):
        alpa_coef = 1 - alpha 
    diff = mde * cr
    return int((norm.ppf(power) + norm.ppf(alpa_coef)) ** 2 * \
           (cr * (1 - cr) + cr * (1 - cr)) \
            / (diff ** 2))

def calc_prop_test_pvalue(variation_num_clicks, variation_size, control_num_clicks, control_size, alternative='larger'):
    """calculates p value for an independent proportional z test
    :param variation_num_clicks: number of clicks in the variation group
    :param variation_size: variation group sample size
    :param control_num_clicks: number of clicks in the control group
    :param control_size: control group sample size
    :param alternative: hypotesis type - larger, smaller, two-sided
    :return: pvalue
    """
    return test_proportions_2indep(variation_num_clicks, 
                                   variation_size, 
                                   control_num_clicks, 
                                   control_size, 
                                   method='wald', 
                                   alternative='larger', 
                                   correction=True).pvalue 