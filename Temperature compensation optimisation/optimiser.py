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
        self.path_to_folder = os.path.abspath(__file__).removesuffix('optimiser.py') + 'Scrubbed data'        
        self.poly_degree = degree

        self.loadData()
    
    def polynomialFitting1(self):
        """
        Dataframe 1 - SB1
        Create multivariate polynomial fit with the measured angle and measured temperature as independent variables (inputs) and the angle difference 
        (reference angle - measured angle) as dependent variable, such that the estimated angle becomes measured angle + angle difference.
        angle_est = angle_meas + diff
        """
        # Generate model of polynomial features
        poly = PolynomialFeatures(degree = self.poly_degree)
        
        # Load data to be used for polyfit
        indep_vars = np.transpose(np.array([self.df1['meas angle'].tolist(), self.df1['meas temp'].tolist()]))
        dep_vars = np.array((self.df1['ref angle'] - self.df1['meas angle']).tolist())
        indep_vars_ = poly.fit_transform(indep_vars)
        print("Feature names: ", poly.get_feature_names_out())
        self.feature_names1 = poly.get_feature_names_out()

        # Create regression object and perform regression
        self.model1 = linear_model.LinearRegression(fit_intercept=True)
        self.model1.fit(indep_vars_, dep_vars)
        self.model1.coef_[0] = self.model1.intercept_
        print("Model coefficients: ", self.model1.coef_)
        return 
    
    def polynomialFitting2(self):
        """
        Dataframe 2 - SB2
        Create multivariate polynomial fit with the measured angle and measured temperature as independent variables (inputs) and the angle difference 
        (reference angle - measured angle) as dependent variable, such that the estimated angle becomes measured angle + angle difference.
        angle_est = angle_meas + diff
        """
        # Generate model of polynomial features
        poly = PolynomialFeatures(degree = self.poly_degree)
        
        # Load data to be used for polyfit
        indep_vars = np.transpose(np.array([self.df2['meas angle'].tolist(), self.df2['meas temp'].tolist()]))
        dep_vars = np.array((self.df2['ref angle'] - self.df2['meas angle']).tolist())
        indep_vars_ = poly.fit_transform(indep_vars) # TO DO: Find out what formula this transforms to.
        print("Feature names: ", poly.get_feature_names_out())
        self.feature_names2 = poly.get_feature_names_out()

        # Create regression object and perform regression
        self.model2 = linear_model.LinearRegression()
        self.model2.fit(indep_vars_, dep_vars)
        print("Model coefficients: ", self.model2.coef_)
        return 

    def polynomialFitting3(self):
        """
        Dataframe 3 - SB3
        Create multivariate polynomial fit with the measured angle and measured temperature as independent variables (inputs) and the angle difference 
        (reference angle - measured angle) as dependent variable, such that the estimated angle becomes measured angle + angle difference.
        angle_est = angle_meas + diff
        """
        # Generate model of polynomial features
        poly = PolynomialFeatures(degree = self.poly_degree)
        
        # Load data to be used for polyfit
        indep_vars = np.transpose(np.array([self.df3['meas angle'].tolist(), self.df3['meas temp'].tolist()]))
        dep_vars = np.array((self.df3['ref angle'] - self.df3['meas angle']).tolist())
        indep_vars_ = poly.fit_transform(indep_vars) # TO DO: Find out what formula this transforms to.
        print("Feature names: ", poly.get_feature_names_out())
        self.feature_names3 = poly.get_feature_names_out()

        # Create regression object and perform regression
        self.model3 = linear_model.LinearRegression()
        self.model3.fit(indep_vars_, dep_vars)
        print("Model coefficients: ", self.model3.coef_)
        return 

    def testFitting1(self):
        """
        Generates a set of 10 random pseudodata points and checks model validity.
        Generate a plot of the polynomial surface, real data points and pseudo testdata for visual inspection.
        """
        n_pseudo = 10 # Number of pseudo datapoints

        # Generate and print pseudodata
        pseudo_meas_angle = [random.uniform(-10., 10.) for i in range(n_pseudo)]
        print("Pseudo measured angle data: ", pseudo_meas_angle)

        pseudo_meas_temp = [random.uniform(-20., 50.) for i in range(n_pseudo)]
        print("Pseudo measured temp data: ", pseudo_meas_temp)

        pseudo_real_angle = []
        for i in range(n_pseudo):
            pseudo_real_angle.append(self.model1.coef_[0]*1 + 
                self.model1.coef_[1]*pseudo_meas_angle[i] + 
                self.model1.coef_[2]*pseudo_meas_temp[i] + 
                self.model1.coef_[3]*pseudo_meas_angle[i]*pseudo_meas_angle[i] +
                self.model1.coef_[4]*pseudo_meas_angle[i]*pseudo_meas_temp[i] + 
                self.model1.coef_[5]*pseudo_meas_temp[i]*pseudo_meas_temp[i])
        
        print("Pseudo real angle data: ", pseudo_real_angle)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # FIRST: Plot surface from model
        x0_mesh = np.arange(-12, 12, 0.01) # Measured angle mesh
        x1_mesh = np.arange(-22, 55, 0.1) # Measured temperature mesh
        x0_mesh, x1_mesh = np.meshgrid(x0_mesh, x1_mesh)
        
        y_mesh = self.model1.coef_[0]*1 + self.model1.coef_[1]*x0_mesh + self.model1.coef_[2]*x1_mesh + self.model1.coef_[3]*x0_mesh*x0_mesh + self.model1.coef_[4]*x0_mesh*x1_mesh + self.model1.coef_[5]*x1_mesh*x1_mesh

        ax.plot_surface(x0_mesh, x1_mesh, y_mesh) # find out transparant surface map

        # SECOND: Plot scattered data
        ax.scatter(self.df1['meas angle'], self.df1['meas temp'], self.df1['ref angle']-self.df1['meas angle'], color='b', label="SB1")

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
    optimiser.polynomialFitting1()
    optimiser.testFitting1()

