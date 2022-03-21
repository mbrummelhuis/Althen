import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

class blockbaxAnalyser():
    def __init__(self, sb_numbers):
        self.sb_numbers = sb_numbers
        self.dfs = []
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cornlowerblue', 'lightcoral', 'chocolate', 'slateblue', 'orange', 'greenyellow']
        self.y_meas = []
        self.y_sigmas = []

        self.slope = None
        self.intercept = None

    def loadData(self):
        for i in range(len(self.sb_numbers)):
            temp_path = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'), 
                str(self.sb_numbers[i]), str(self.sb_numbers[i]) + '_blockbax_temp.csv')
            y_path = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'), 
                str(self.sb_numbers[i]), str(self.sb_numbers[i]) + '_blockbax_y.csv')
            
            names = ['Series', 'Date', 'Value']
            temp_df = pd.read_csv(temp_path, delimiter=',"', header=1, engine='python', names=names).replace('"','', regex=True)
            temp_df['Date'] = pd.to_datetime(temp_df["Date"], format = "%Y-%m-%d %H:%M:%S")
            y_df = pd.read_csv(y_path, delimiter=',"', header=1, engine='python', names=names).replace('"','', regex=True)

            temp_df['Value'] = pd.to_numeric(temp_df['Value'], downcast="float")
            y_df['Value'] = pd.to_numeric(y_df['Value'], downcast="float")

            self.y_meas.append(y_df["Value"].mean())
            self.y_sigmas.append(y_df["Value"].std())

            data = [temp_df["Date"], temp_df["Value"], y_df["Value"], (y_df["Value"]-self.y_meas[i]), (y_df["Value"]-self.y_meas[i])/self.y_sigmas[i]]
            headers = ['Datetime', 'Temperature', 'Y value', 'Y value (half norm)', 'Y value (norm)']

            self.dfs.append(pd.concat(data, axis=1, keys=headers))

    
    def plotTempAngle(self):
        for i in range(len(self.sb_numbers)):
            plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Y value'], label=str(self.sb_numbers[i]))

        if self.slope != None:
            x = np.linspace(0,25,1000)
            y = self.slope * x + self.intercept
            plt.plot(x,y)
        
        plt.legend()
        plt.title("Temperature vs Y value")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Measured Angle Y')
        plt.show()

    def plotTempHalfNormAngle(self):
        for i in range(len(self.sb_numbers)):
            plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Y value (half norm)'], label=str(self.sb_numbers[i]))
        
        if self.slope != None:
            x = np.linspace(0,20,1000)
            y = self.slope * x + self.intercept
            plt.plot(x,y)
        
        plt.legend()
        plt.title("Temperature vs Y value")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Normalised Angle Y')
        plt.show()

    def plotTempNormAngle(self):
        for i in range(len(self.sb_numbers)):
            plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Y value (norm)'], label=str(self.sb_numbers[i]))
        
        if self.slope != None:
            x = np.linspace(0,20,1000)
            y = self.slope * x + self.intercept
            plt.plot(x,y)
        
        plt.legend()
        plt.title("Temperature vs Y value")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Normalised Angle Y')
        plt.show()
    
    def trendline(self):
        templist = []
        ylist = []
        # Append all temperature and y data in separate lists
        for i in range(len(self.sb_numbers)):
            templist += self.dfs[i]['Temperature'].tolist()
            ylist += self.dfs[i]['Y value'].tolist()

        temparray = np.array(templist)
        yarray = np.array(ylist)

        self.slope, self.intercept, r_value, p_value, std_err = linregress(temparray, yarray)
        print("slope: %f, intercept: %f" % (self.slope, self.intercept))
        print("R-squared: %f" % r_value**2)

    def trendlineHalfNorm(self):
        templist = []
        ylist = []
        # Append all temperature and y data in separate lists
        for i in range(len(self.sb_numbers)):
            templist += self.dfs[i]['Temperature'].tolist()
            ylist += self.dfs[i]['Y value (half norm)'].tolist()

        temparray = np.array(templist)
        yarray = np.array(ylist)

        self.slope, self.intercept, r_value, p_value, std_err = linregress(temparray, yarray)
        print("slope: %f, intercept: %f" % (self.slope, self.intercept))
        print("R-squared: %f" % r_value**2)

    def trendlineNorm(self):
        templist = []
        ylist = []
        # Append all temperature and y data in separate lists
        for i in range(len(self.sb_numbers)):
            templist += self.dfs[i]['Temperature'].tolist()
            ylist += self.dfs[i]['Y value (norm)'].tolist()

        temparray = np.array(templist)
        yarray = np.array(ylist)

        self.slope, self.intercept, r_value, p_value, std_err = linregress(temparray, yarray)
        print("slope: %f, intercept: %f" % (self.slope, self.intercept))
        print("R-squared: %f" % r_value**2)

if __name__=="__main__":
    #sb_numbers = [141419, 141422, 141425, 141427, 141430, 141431, 141433]
    sbs = [148040, 148054, 148062, 148068, 148070, 148074, 148077, 148084, 148089, 148094, 148101, 148102]

    for sb in sbs:    
        analyser = blockbaxAnalyser([sb])
        analyser.loadData()
        print("-------------------"+str(sb))
        print(analyser.y_meas)
        print(analyser.y_sigmas)

        #print("Not normalised --------------------------")
        #analyser.trendline()
        #analyser.plotTempAngle()

        #print("Fully normalised --------------------------")
        analyser.trendlineNorm()
        analyser.plotTempNormAngle()

        #print("Half normalised --------------------------")
        #analyser.trendlineHalfNorm()
        #analyser.plotTempHalfNormAngle()
