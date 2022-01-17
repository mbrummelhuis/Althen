import pandas as pd
from comparator import Comparator
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
import os
import matplotlib.pyplot as plt

class Optimiser(Comparator):
    def __init__(self, degree):
        self.path_to_folder = os.path.abspath(__file__).removesuffix('optimiser.py') + 'Scrubbed data'        
        self.poly_degree = degree

        self.loadData()


    def polynomial(x, coeffs):
        """ 
        Returns a polynomial for ``x`` values for the ``coeffs`` provided.

        The coefficients must be in ascending order (``x**0`` to ``x**o``).
        """
        o = len(coeffs)
        y_vals = []
        for n in range(len(x)):
            y = 0
            p = x[n]
            for i in range(o):
                y += coeffs[i]*p**i
            y_vals.append(y)
        return y_vals
    
    def polynomialFitting(self):
        """
        Create multivariate polynomial fit with the measured angle and measured temperature as independent variables (inputs) and the angle difference 
        (reference angle - measured angle) as dependent variable, such that the estimated angle becomes measured angle + angle difference.
        angle_est = angle_meas + diff
        """
        # Generate model of polynomial features
        poly = PolynomialFeatures(degree = self.poly_degree)
        
        # Load data to be used for polyfit
        indep_vars = np.transpose(np.array([self.df1['meas angle'].tolist(), self.df1['meas temp'].tolist()]))
        dep_vars = np.array((self.df1['ref angle'] - self.df1['meas angle']).tolist())
        indep_vars_ = poly.fit_transform(indep_vars) # TO DO: Find out what formula this transforms to.
        print("Coefficient terms: ", poly.get_feature_names_out())
        print(indep_vars)

        # Create regression object and perform regression
        self.model = linear_model.LinearRegression()
        self.model.fit(indep_vars_, dep_vars)
        return 

    def testFitting(self):
        # Plot fitted polynomial vs data for visual inspection and give a set of randomly generated pseudodata to check model validity.

        return False
    
    def printModel(self):
        print("Model coefficients: ", self.model.coef_)
        print("")
        return

if __name__ == "__main__":
    optimiser = Optimiser(degree = 2)

    optimiser.polynomialFitting()
    optimiser.printModel()

