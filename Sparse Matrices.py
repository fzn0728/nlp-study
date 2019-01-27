# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:08:31 2019

@author: Chandler
"""

class SparseMetrices (object):
    """
    Initiate the Sparse Metrices
    """
    def __init__(self):
        self.row = []
        self.column = []
        self.value = []

        
class SparseMetricsOperation (object): 
    """
    Perform Sparce Metrices Operation
    """
    def SparseMetricesSum(self, metricesA, metricesB):
        length = set(len(metricesA.value),len(metricesB.value))
        #for i in range(len())
        
        return length
        
        
def main():       
    metricesA = SparseMetrices()
    metricesA.row = [1,2,3]
    metricesA.column = [1,2,3]
    metricesA.value = [1,1,1]
    
    metricesB = SparseMetrices()
    metricesB.row = [1,2,3]
    metricesB.column = [1,2,3]
    metricesB.value = [1,1,1]
    