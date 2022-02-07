from tkinter import Y
import pandas as pd
from comparator import Comparator
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import random

pd.options.mode.chained_assignment = None  # default='warn'

class Optimiser(Comparator):
    def __init__(self, degree):
        super(Optimiser, self).__init__(names=['SB1', 'SB2', 'SB3'])
        self.path_to_folder = os.path.abspath(__file__).removesuffix('optimiser.py') + 'Scrubbed data'        
        self.poly_degree = degree
        self.feature_names = []
        self.models = []

        self.loadDatav2()
    
    def polynomialFitting(self):
        """
        Create multivariate polynomial fit with the measured angle and measured temperature as independent variables (inputs) and the angle difference 
        (reference angle - measured angle) as dependent variable, such that the estimated angle becomes measured angle + angle difference.
        angle_est = angle_meas + diff
        """
        for i in range(len(self.dfs)):
            print("Model for dataframe ", i)
            # Generate model of polynomial features
            poly = PolynomialFeatures(degree = self.poly_degree)
            
            # Load data to be used for polyfit
            indep_vars = np.transpose(np.array([self.dfs[i]['meas angle'].tolist(), self.dfs[i]['meas temp'].tolist()]))
            dep_vars = np.array((self.dfs[i]['ref angle'] - self.dfs[i]['meas angle']).tolist())
            indep_vars_ = poly.fit_transform(indep_vars)
            print("Feature names: ", poly.get_feature_names_out())
            self.feature_names.append(poly.get_feature_names_out())

            # Create regression object and perform regression
            self.models.append(linear_model.LinearRegression(fit_intercept=True))
            self.models[i].fit(indep_vars_, dep_vars)
            self.models[i].coef_[0] = self.models[i].intercept_
            print("Model coefficients: ", self.models[i].coef_)
        return 

    def testFitting(self):
        """
        Generates a set of 10 random pseudodata points and checks model validity.
        Generate a plot of the polynomial surface, real data points and pseudo testdata for visual inspection.
        """
        for i in range(len(self.dfs)):
            n_pseudo = 10 # Number of pseudo datapoints

            # Generate and print pseudodata
            pseudo_meas_angle = [random.uniform(-10., 10.) for i in range(n_pseudo)]
            print("Pseudo measured angle data: ", pseudo_meas_angle)

            pseudo_meas_temp = [random.uniform(-20., 50.) for i in range(n_pseudo)]
            print("Pseudo measured temp data: ", pseudo_meas_temp)

            pseudo_real_angle = []
            for n in range(n_pseudo):
                pseudo_real_angle.append(self.models[i].coef_[0]*1 + 
                    self.models[i].coef_[1]*pseudo_meas_angle[n] + 
                    self.models[i].coef_[2]*pseudo_meas_temp[n] + 
                    self.models[i].coef_[3]*pseudo_meas_angle[n]*pseudo_meas_angle[n] +
                    self.models[i].coef_[4]*pseudo_meas_angle[n]*pseudo_meas_temp[n] + 
                    self.models[i].coef_[5]*pseudo_meas_temp[n]*pseudo_meas_temp[n])
            
            print("Pseudo real angle data: ", pseudo_real_angle)

            # Plot
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            # FIRST: Plot surface from model
            x0_mesh = np.arange(-12, 12, 0.01) # Measured angle mesh
            x1_mesh = np.arange(-22, 55, 0.1) # Measured temperature mesh
            x0_mesh, x1_mesh = np.meshgrid(x0_mesh, x1_mesh)
            
            y_mesh = self.models[i].coef_[0]*1 + self.models[i].coef_[1]*x0_mesh + self.models[i].coef_[2]*x1_mesh + self.models[i].coef_[3]*x0_mesh*x0_mesh + self.models[i].coef_[4]*x0_mesh*x1_mesh + self.models[i].coef_[5]*x1_mesh*x1_mesh

            ax.plot_surface(x0_mesh, x1_mesh, y_mesh) # find out transparant surface map

            # SECOND: Plot scattered data
            ax.scatter(self.dfs[i]['meas angle'], self.dfs[i]['meas temp'], self.dfs[i]['ref angle']-self.dfs[i]['meas angle'], color=self.colours[i], label="SB"+str(i+1))

            # THIRD: Plot pseudo data
            ax.scatter(pseudo_meas_angle, pseudo_meas_temp, pseudo_real_angle, color='m', label='Pseudo')

            # Cleanup
            ax.set_xlabel("Measured angle")
            ax.set_ylabel("Measured temperature")
            ax.set_zlabel("Ref - meas angle")
            ax.legend()
            plt.show()
        return
    
if __name__ == "__main__":
    optimiser = Optimiser(degree = 2)
    optimiser.polynomialFitting()
    optimiser.testFitting()

