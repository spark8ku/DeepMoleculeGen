from abc import ABCMeta, abstractmethod

import numpy as np

__all__ = ['Conditional', 'Delimited']

class Conditional(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class Delimited(Conditional):

    def __init__(self, d='\t'):
        self.d = d

    def __call__(self, line):
        line = line.strip('\n').strip('\r')
        line = line.split(self.d)

        smiles = line[0]
        smiles_sol = line[1]
        c = np.array([float(c_i) for c_i in line[2:9]], dtype=np.float32) ###### Condition Num
        return smiles,smiles_sol,c

