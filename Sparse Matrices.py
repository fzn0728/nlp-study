# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:08:31 2019

@author: Chandler
"""

class SparseMatrices (object):
    """
    Initiate the Sparse Matrices
    """
    def __init__(self):
        self.row = []
        self.column = []
        self.value = []

        
class SparseMatricesOperation (object): 
    """
    Perform Sparce Matrices Operation
    
    # Example:
    Row Col Val      Row Col Val
    1   2   10       1   1   2
    1   3   12       1   2   5
    2   1   1        2   2   1
    2   3   2        3   1   8    
    
    Transpose of second matrix:
    
    Row Col Val      Row Col Val
    1   2   10       1   1   2
    1   3   12       1   3   8
    2   1   1        2   1   5
    2   3   2        2   2   1
    
    Summation of multiplied values:
    
    result[1][1] = A[1][3]*B[1][3] = 12*8 = 96
    result[1][2] = A[1][2]*B[2][2] = 10*1 = 10
    result[2][1] = A[2][1]*B[1][1] + A[2][3]*B[1][3] = 2*1 + 2*8 = 18
    result[2][2] = A[2][1]*B[2][1] = 1*5 = 5
    
    Any other element cannot be obtained 
    by any combination of row in 
    Matrix A and Row in Matrix B.    
    
    Hence the final resultant matrix will be:
     
    Row Col Val 
    1   1   96 
    1   2   10 
    2   1   18  
    2   2   5      
    """
    
    def SparsematricesSum(self, matricesA, matricesB):
        """
        Apply Matrices Sum
        """
        # Do sum
        length = set(len(matricesA.value),len(matricesB.value))
        #for i in range(length):
            
        
        # insert remaining elements
        return 
    
    def SparseMatricesTranspose (self):
        """
        Transpose the matrices
        """
        
        
        return
    
    def SparseMatricesMultiply (self):
        """
        Take matrices A and transposed matrices B and apply multipication
        """
        # Take matrices B and transpose
        
        # Apply multiply
        
        return
    
        
        
def main():    
    # Initialize
    matricesA = SparseMatrices()
    matricesA.row = [1,2,3]
    matricesA.column = [1,2,3]
    matricesA.value = [1,1,1]
    
    matricesB = SparseMatrices()
    matricesB.row = [1,2,3]
    matricesB.column = [1,2,3]
    matricesB.value = [1,1,1]
    
    # Calculation
    smo = SparseMatricesOperation()
    smo.SparseMatricesSum(matricesA,matricesB)
    smo.SparseMatricesMultiply(matricesA,matricesB)
    