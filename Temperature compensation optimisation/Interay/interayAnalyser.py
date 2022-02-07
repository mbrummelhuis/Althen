import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

class interayAnalyser:
    def __init__(self, sb_numbers, degree=2):
        self.sb_numbers = sb_numbers
        self.filenames = []
        self.colours = ['b', 'r', 'g']
        self.paths = []
        self.dfs = []
        self.poly_degree = degree

        for index in range(len(self.sb_numbers)):
            self.filenames.append('SB_' + str(self.sb_numbers[index]) + '_Interay_alldata_cleaned_balanced_offset.xlsx')
            self.paths.append(os.path.join(os.path.abspath(__file__).removesuffix('interayAnalyser.py'), 
                str(self.sb_numbers[index]), self.filenames[index]))

    def loadData(self):
        for i in range(len(self.sb_numbers)):
            self.dfs.append(pd.read_excel(self.paths[i]))

            # Rename columns to shorter, more sensible names including units where known
            self.dfs[i] = self.dfs[i].rename({'SmartBrick ('+str(self.sb_numbers[i])+') Battery': 'Battery (V)', 
                'SmartBrick ('+str(self.sb_numbers[i])+') Cause': 'Trigger',
                'SmartBrick ('+str(self.sb_numbers[i])+') Humidity': 'Humidity (%)',
                'SmartBrick ('+str(self.sb_numbers[i])+') Signal Strength':'Signal (dB)',
                'SmartBrick ('+str(self.sb_numbers[i])+') Temperature':'Temp (C)',
                'SmartBrick ('+str(self.sb_numbers[i])+') X':'X',
                'SmartBrick ('+str(self.sb_numbers[i])+') X ADC':'X ADC',
                'SmartBrick ('+str(self.sb_numbers[i])+') Y':'Y (off)',
                'SmartBrick ('+str(self.sb_numbers[i])+') Y ADC':'Y ADC', 
                'Reference angle':'ref angle',
                'Angle setting':'Angle setting',
                'Temperature setting':'Temp setting',
                'Y w/o offset':'Y'}, axis=1)
            self.dfs[i]['Time'] = pd.to_datetime(self.dfs[i]["Time"], format = "%Y-%m-%d %H:%M:%S")
            self.dfs[i]['Y'] = -1*self.dfs[i]['Y']
            print(self.dfs[i])
        return
    
    def polyFitSingle(self):
        """
        Create multivariate polynomial fit with the measured angle and measured temperature as independent variables (inputs) and the angle difference 
        (reference angle - measured angle) as dependent variable, such that the estimated angle becomes measured angle + angle difference.
        angle_est = angle_meas + diff
        """
        self.models = []
        for i in range(len(self.sb_numbers)):
            print("PolyFit for Interay Smartbrick with number", self.sb_numbers[i])
            poly = PolynomialFeatures(degree = self.poly_degree)
            indep_vars = np.transpose(np.array([self.dfs[i]['Y'].tolist(), self.dfs[i]['Temp (C)'].tolist()]))
            dep_vars = np.array((self.dfs[i]['ref angle'] - self.dfs[i]['Y']).tolist())
            indep_vars_ = poly.fit_transform(indep_vars)
            print("Feature names: ", poly.get_feature_names_out())
            self.feature_names = poly.get_feature_names_out()

            # Create regression object and perform regression
            self.models.append(linear_model.LinearRegression(fit_intercept=True))
            self.models[i].fit(indep_vars_, dep_vars)
            self.models[i].coef_[0] = self.models[i].intercept_
            print("Model coefficients: ", self.models[i].coef_)

        return 

    def polyFitDouble(self):
        ylist = []
        templist = []
        reflist = []
        print("Polynomial fit with regression to data of both Smartbricks")

        for i in range(len(self.sb_numbers)):
            poly = PolynomialFeatures(degree = self.poly_degree)
            ylist = ylist + self.dfs[i]['Y'].tolist()
            templist = templist + self.dfs[i]['Temp (C)'].tolist()

            reflist = reflist +((self.dfs[i]['ref angle'] - self.dfs[i]['Y']).tolist())
        indep_vars = np.transpose(np.array([ylist, templist]))
        dep_vars = np.array(reflist)

        indep_vars_ = poly.fit_transform(indep_vars)
        print("Feature names: ", poly.get_feature_names_out())
        self.feature_names = poly.get_feature_names_out()

        # Create regression object and perform regression
        self.modeldouble = linear_model.LinearRegression(fit_intercept=True)
        self.modeldouble.fit(indep_vars_, dep_vars)
        self.modeldouble.coef_[0] = self.modeldouble.intercept_
        print("Model coefficients: ", self.modeldouble.coef_)
        pass
        
    def plotTempX(self):
        """
        Creates a scatterplot of the different smartbrick's measured temperature against X angle
        """
        for i in range(len(self.dfs)):
            plt.scatter(self.dfs[i]['Temp (C)'], self.dfs[i]['X'], color=self.colours[i], label=str(self.sb_numbers[i]))

        plt.grid()
        plt.legend()
        plt.title("Temperature dependence of angle X")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Measured Angle X')
        plt.show()

    def plotTempY(self):
        """
        Creates a scatterplot of the different smartbrick's measured temperature against Y angle
        """
        for i in range(len(self.dfs)):
            plt.scatter(self.dfs[i]['Temp (C)'], self.dfs[i]['Y'], color=self.colours[i], label=str(self.sb_numbers[i]))

        plt.grid()
        plt.legend()
        plt.title("Temperature dependence of angle Y")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Measured Angle Y')
        plt.show()
    
    def plotModel(self):
        for i in range(len(self.sb_numbers)):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            # FIRST: Plot surface from model
            x0_mesh = np.arange(-12, 12, 0.01) # Measured angle mesh
            x1_mesh = np.arange(-22, 55, 0.1) # Measured temperature mesh
            x0_mesh, x1_mesh = np.meshgrid(x0_mesh, x1_mesh)
            
            y_mesh = self.models[i].coef_[0]*1 + self.models[i].coef_[1]*x0_mesh + self.models[i].coef_[2]*x1_mesh + self.models[i].coef_[3]*x0_mesh*x0_mesh + self.models[i].coef_[4]*x0_mesh*x1_mesh + self.models[i].coef_[5]*x1_mesh*x1_mesh

            ax.plot_surface(x0_mesh, x1_mesh, y_mesh) # find out transparant surface map

            # SECOND: Plot scattered data
            ax.scatter(self.dfs[i]['Y'], self.dfs[i]['Temp (C)'], self.dfs[i]['ref angle']-self.dfs[i]['Y'], color=self.colours[i], label="SB"+str(i+1))

            # Cleanup
            ax.set_xlabel("Measured angle")
            ax.set_ylabel("Measured temperature")
            ax.set_zlabel("Ref - meas angle")
            ax.legend()
            plt.show()

    def plotModelDouble(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # FIRST: Plot surface from model
        x0_mesh = np.arange(-12, 12, 0.01) # Measured angle mesh
        x1_mesh = np.arange(-22, 55, 0.1) # Measured temperature mesh
        x0_mesh, x1_mesh = np.meshgrid(x0_mesh, x1_mesh)
        
        y_mesh = self.modeldouble.coef_[0]*1 + self.modeldouble.coef_[1]*x0_mesh + self.modeldouble.coef_[2]*x1_mesh + self.modeldouble.coef_[3]*x0_mesh*x0_mesh + self.modeldouble.coef_[4]*x0_mesh*x1_mesh + self.modeldouble.coef_[5]*x1_mesh*x1_mesh

        ax.plot_surface(x0_mesh, x1_mesh, y_mesh) # find out transparant surface map

        # SECOND: Plot scattered data
        for i in range(len(self.sb_numbers)):
            ax.scatter(self.dfs[i]['Y'], self.dfs[i]['Temp (C)'], self.dfs[i]['ref angle']-self.dfs[i]['Y'], color=self.colours[i], label="SB"+str(i+1))

        # Cleanup
        ax.set_xlabel("Measured angle")
        ax.set_ylabel("Measured temperature")
        ax.set_zlabel("Ref - meas angle")
        ax.legend()
        plt.show()

if __name__ == '__main__':
    SB_numbers = [141421, 141442]
    analyser = interayAnalyser(sb_numbers=SB_numbers)
    analyser.loadData()
    analyser.polyFitSingle()
    analyser.plotModel()
    analyser.polyFitDouble()
    analyser.plotModelDouble()


