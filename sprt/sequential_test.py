import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import scipy.stats as stats

class Sequential_test:
    """Wald sequential test 
    :param control: list of observations from Control group (sorted by date)
    :param variation: list of observations from Variation group (sorted by date)
    :param mde: MDE
    :param alpha: alpa 
    :param beta: beta
    """
    def __init__(self, control, variation, mde=0.05, alpha=0.05, beta=0.2):
        self.control = np.array(control)
        self.variation = np.array(variation)
        self.mde = mde
        self.alpha = alpha
        self.beta = beta
        
    def _calculate_z(self, control, variation):
        """z statistics calculation"""
        mean_control = np.mean(control)
        std_control = np.std(control)
        lower_bound = np.log(self.beta / (1 - self.alpha))
        upper_bound = np.log((1 - self.beta) / self.alpha)
        min_len = min([len(control), len(variation)])
        delta = variation - control
        delta_var = delta.std() ** 2
        pdf_one_values = np.exp(-(delta - 0) ** 2 / (2 * delta_var))
        pdf_two_values = np.exp(-(delta - mean_control * self.mde) ** 2 / (2 * delta_var))
        z = np.sum(np.log(pdf_two_values / pdf_one_values))
        return z
    
    @property
    def calculate(self):
        """test result for the whole list of observations"""
        z = self._calculate_z(self.control, self.variation)
        lower_bound = np.log(self.beta / (1 - self.alpha))
        upper_bound = np.log((1 - self.beta) / self.alpha)
        if len(self.control) < 50: z = 0
        if z < lower_bound:
            print('Reject H1')
        elif z > upper_bound:
            print('Reject H0')
        else: 
            print('Not enough data')
            
    @property
    def calculate_(self):
        """test result for the whole list of observations"""
        z = self._calculate_z(self.control, self.variation)
        lower_bound = np.log(self.beta / (1 - self.alpha))
        upper_bound = np.log((1 - self.beta) / self.alpha)
        if len(self.control) < 50: z = 0
        if z < lower_bound:
            return 0
        elif z > upper_bound:
            return 1
        else: 
            return 0.5
            
    @property
    def calculate_history(self):
        """historical test results"""
        history_z = []
        for i in range(0, len(self.control), 500):
            if i < 50: 
                z_temp = 0
            else:
                z_temp = self._calculate_z(self.control[:i], self.variation[:i])
            history_z.append(z_temp)
        lower_bound = np.log(self.beta / (1 - self.alpha))
        upper_bound = np.log((1 - self.beta) / self.alpha)
        return pd.DataFrame({'z':history_z, 'lower_bound':lower_bound, 'upper_bound':upper_bound})
    
    @property
    def plot(self):
        """plots the statistics and bounds"""
        df = self.calculate_history
        sns.lineplot(data=df.z)
        sns.lineplot(data=df.lower_bound)
        sns.lineplot(data=df.upper_bound)