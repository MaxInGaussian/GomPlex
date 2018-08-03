################################################################################
#  Github: https://github.com/MaxInGaussian/GomPlex
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np

__all__ = [
    "Scaler"
]

class Scaler(object):

    scalers = [
        "minmax",
        "normal",
    ]
    
    def __init__(self, scaler, matrix):
        assert scaler in self.scalers, "Invalid scaler!"
        self.scaler = scaler
        getattr(self, self.scaler+'_init')(np.complex_(matrix)+0j)

    def eval(self, matrix, inv=False):
        if(inv):            
            return  getattr(self, self.scaler+'_inv')(np.complex_(matrix)+0j)
        return getattr(self, self.scaler)(np.complex_(matrix)+0j)

    def minmax_init(self, matrix):
        self._r_min_ = np.min(matrix.real, axis=0)
        self._r_max_ = np.max(matrix.real, axis=0)
        if(np.any(self._r_min_ == self._r_max_)):
            self._r_max_ += .5
            self._r_min_ -= .5
        if(np.any(np.iscomplex(matrix))):
            self._i_min_ = np.min(matrix.imag, axis=0)
            self._i_max_ = np.max(matrix.imag, axis=0)
            if(np.any(self._i_min_ == self._i_max_)):
                self._i_max_ += .5
                self._i_min_ -= .5

    def minmax(self, matrix):
        res = (matrix.real-self._r_min_)/(self._r_max_-self._r_min_)-.5
        if(np.any(np.iscomplex(matrix))):
            res = res+0j
            res += ((matrix.imag-self._i_min_)/(self._i_max_-self._i_min_)-.5)*1j
        return res

    def minmax_inv(self, matrix):
        res = (matrix.real+.5)*(self._r_max_-self._r_min_)+self._r_min_
        if(np.any(np.iscomplex(matrix))):
            res = res+0j
            res += ((matrix.imag+.5)*(self._i_max_-self._i_min_)+self._i_min_)*1j
        return res

    def normal_init(self, matrix):
        self._r_mu_ = np.mean(matrix.real, axis=0)
        self._r_std_ = np.std(matrix.real, axis=0)
        if(np.any(self._r_std_ == 0)):
            self._r_std_ += 1e-6
        if(np.any(np.iscomplex(matrix))):
            self._i_mu_ = np.mean(matrix.imag, axis=0)
            self._i_std_ = np.std(matrix.imag, axis=0)
            if(np.any(self._i_std_ == 0)):
                self._i_std_ += 1e-6

    def normal(self, matrix):
        res = (matrix.real-self._r_mu_)/self._r_std_
        if(np.any(np.iscomplex(matrix))):
            res = res+0j
            res += (matrix.imag-self._i_mu_)/self._i_std_*1j
        return res

    def normal_inv(self, matrix):
        res = matrix.real*self._r_std_+self._r_mu_
        if(np.any(np.iscomplex(matrix))):
            res = res+0j
            res += (matrix.imag*self._i_std_+self._i_mu_)*1j
        return res
    
