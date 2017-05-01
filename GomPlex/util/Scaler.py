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
        self.__r_min__ = np.min(matrix.real, axis=0)
        self.__r_max__ = np.max(matrix.real, axis=0)
        if(np.any(self.__r_min__ == self.__r_max__)):
            self.__r_max__ += .5
            self.__r_min__ -= .5
        self.__i_min__ = np.min(matrix.imag, axis=0)
        self.__i_max__ = np.max(matrix.imag, axis=0)
        if(np.any(self.__i_min__ == self.__i_max__)):
            self.__i_max__ += .5
            self.__i_min__ -= .5

    def minmax(self, matrix):
        return (matrix.real-self.__r_min__)/(self.__r_max__-self.__r_min__)-.5+\
            ((matrix.imag-self.__i_min__)/(self.__i_max__-self.__i_min__)-.5)*1j

    def minmax_inv(self, matrix):
        return (matrix.real+.5)*(self.__r_max__-self.__r_min__)+self.__r_min__+\
            ((matrix.imag+.5)*(self.__i_max__-self.__i_min__)+self.__i_min__)*1j

    def normal_init(self, matrix):
        self.__r_mu__ = np.mean(matrix.real, axis=0)
        self.__r_std__ = np.std(matrix.real, axis=0)
        if(np.any(self.__r_std__ == 0)):
            self.__r_std__ += 1e-6
        self.__i_mu__ = np.mean(matrix.imag, axis=0)
        self.__i_std__ = np.std(matrix.imag, axis=0)
        if(np.any(self.__i_std__ == 0)):
            self.__i_std__ += 1e-6

    def normal(self, matrix):
        return (matrix.real-self.__r_mu__)/self.__r_std__+\
            (matrix.imag-self.__i_mu__)/self.__i_std__*1j

    def normal_inv(self, matrix):
        return matrix.real*self.__r_std__+self.__r_mu__+\
            (matrix.imag*self.__i_std__+self.__i_mu__)*1j
    
