################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np

__all__ = [
    "Metric"
]

class Metric(object):
    
    metrics = [
        "mse",
        "nmse",
        "mae",
        "nlpd",
        "nlpd_mse",
        "nlpd_nmse",
        "nlml"
    ]
    
    def __init__(self, metric, gp=None):
        assert metric in self.metrics, "Invalid metric!"
        self.metric = metric  
        self.gp = gp

    def eval(self, target, mu_pred, std_pred):
        return getattr(self, self.metric)(target, mu_pred, std_pred)

    def mse(self, target, mu_pred, std_pred):        
        mse = np.mean(np.absolute(target-mu_pred)**2)
        return mse

    def nmse(self, target, mu_pred, std_pred):
        mse = self.mse(target, mu_pred, std_pred)
        nmse = mse/np.var(np.absolute(target))
        return nmse

    def mae(self, target, mu_pred, std_pred):
        mae = np.mean(np.absolute(target-mu_pred))
        return mae

    def nlpd(self, target, mu_pred, std_pred):        
        nlpd = np.mean(((target-mu_pred)/std_pred)**2+2*np.log(std_pred))
        nlpd = 0.5*(np.log(2*np.pi)+nlpd)
        return np.absolute(nlpd)

    def nlpd_mse(self, target, mu_pred, std_pred):        
        return self.mse(target, mu_pred, std_pred)-\
            np.exp(-self.nlpd(target, mu_pred, std_pred))

    def nlpd_nmse(self, target, mu_pred, std_pred):        
        return self.nmse(target, mu_pred, std_pred)-\
            np.exp(-self.nlpd(target, mu_pred, std_pred))
    
    def nlml(self, target, mu_pred, std_pred):
        if(self.gp.mean_only):
            return self.mse(target, mu_pred, std_pred)
        noise = self.gp.noise_real+self.gp.noise_imag*1j
        goodness_of_fit = (target.conj().T.dot(target-mu_pred))/noise
        covariance_penalty = np.sum(np.log(np.diagonal(self.gp.T)))
        noise_penalty = (self.gp.N-self.gp.M)*np.log(noise)
        nlml = goodness_of_fit+covariance_penalty+noise_penalty
        return np.absolute(nlml[0, 0])
