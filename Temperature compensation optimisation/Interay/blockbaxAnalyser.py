import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

class blockbaxAnalyser():
    def __init__(self, sb_numbers, plot_from_date, plot_till_date):
        self.sb_numbers = sb_numbers
        self.dfs = []
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cornlowerblue', 'lightcoral', 'chocolate', 'slateblue', 'orange', 'greenyellow']
        self.y_means = []
        self.y_sigmas = []

        self.plot_from_date = plot_from_date
        self.plot_till_date = plot_till_date

        self.slope = None
        self.intercept = None

        # Trendline settings (min and max temperature)
        self.tl_min = -20
        self.tl_max =  50

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

            # Delete before and after
            data = [temp_df["Date"], temp_df["Value"], y_df["Value"]]
            headers = ['Datetime', 'Temperature', 'Y value']
            data = pd.concat(data, axis=1, keys=headers)

            data["Datetime"] = pd.to_datetime(data["Datetime"], format = "%Y-%m-%d %H:%M:%S")
            data.sort_index(axis=0, ascending=False)
            data = data.set_index("Datetime", drop=False)
            delete_before = pd.Timestamp(self.plot_from_date) # Year, month, date
            delete_after = pd.Timestamp(self.plot_till_date)
            data = data.loc[(data.index > delete_before)]
            data = data.loc[(data.index < delete_after)]

            # Calculate means and stdevs            
            self.y_means.append(data["Y value"].mean())
            self.y_sigmas.append(data["Y value"].std())

            # Construct extended dataset
            ext_data = [data["Datetime"], data["Temperature"], data["Y value"], (data["Y value"]-self.y_means[i]), (data["Y value"]-self.y_means[i])/self.y_sigmas[i]]
            ext_headers = ['Datetime', 'Temperature', 'Y value', 'Y value (half norm)', 'Y value (norm)']

            # Append dataframe to object attribute
            self.dfs.append(pd.concat(ext_data, axis=1, keys=ext_headers))

    
    def plotTempAngle(self):
        """
        Plots temperature on x-axis and angle on y-axis of all smartbricks.
        If trendline has been calculated, will plot trendline
        """
        for i in range(len(self.sb_numbers)):
            plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Y value'], label=str(self.sb_numbers[i]))

        # Plot trendline
        if self.slope != None:
            x = np.linspace(self.tl_min,self.tl_max,1000)
            y = self.slope * x + self.intercept
            plt.plot(x,y)
        
        plt.legend()
        plt.title("Temperature vs Y value")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Measured Angle Y')
        plt.show()

    def plotTempHalfNormAngle(self):
        """
        Plots temperature on x-axis and half normalised (w.r.t. mean) angle on y-axis of all smartbricks.
        If trendline has been calculated, will plot trendline        
        """
        for i in range(len(self.sb_numbers)):
            plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Y value (half norm)'], label=str(self.sb_numbers[i]))
        
        if self.slope != None:
            x = np.linspace(self.tl_min,self.tl_max,1000)
            y = self.slope * x + self.intercept
            plt.plot(x,y)
        
        plt.legend()
        plt.title("Temperature vs Y value")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Normalised Angle Y')
        plt.show()

    def plotTempNormAngle(self):
        """
        Plots temperature on x-axis and fully normalised (w.r.t. mean and stdev) angle on y-axis of all smartbricks.
        If trendline has been calculated, will plot trendline           
        """
        for i in range(len(self.sb_numbers)):
            plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Y value (norm)'], label=str(self.sb_numbers[i]))
        
        # Plot trendline
        if self.slope != None:
            x = np.linspace(self.tl_min,self.tl_max,1000)
            y = self.slope * x + self.intercept
            plt.plot(x,y)
        
        plt.legend()
        plt.title("Temperature vs Y value")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Normalised Angle Y')
        plt.show()
    
    def trendline(self):
        """
        Calculates the slope, intercept and r-squared for a trendline of a set.
        """
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
        """
        Calculates the slope, intercept and r-squared for a trendline of a set with angles normalised w.r.t. mean.
        """
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
        """
        Calculates the slope, intercept and r-squared for a trendline of a set with angles normalised w.r.t. mean and stdev.
        """
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
    #sbs = [148040, 148054, 148062, 148068, 148070, 148074, 148077, 148084, 148089, 148094, 148101, 148102]

    plot_from_date = "2022-03-21 18:00:00" # put in date string with format YYYY-MM-DD HH:MM:SS
    plot_till_date = "2022-03-24 01:30:00"

    sbs = [148088, 148097, 148098, 148099, 148105, 148106, 148107]

    analyser = blockbaxAnalyser(sbs, plot_from_date, plot_till_date)
    analyser.loadData()
    analyser.plotTempNormAngle()


    for sb in sbs:    
        analyser = blockbaxAnalyser([sb], plot_from_date, plot_till_date)
        analyser.loadData()
        print("-------------------"+str(sb))
        print("Means: ", analyser.y_means)
        print("Stdev: ", analyser.y_sigmas)

        #print("Not normalised --------------------------")
        #analyser.trendline()
        #analyser.plotTempAngle()

        #print("Fully normalised --------------------------")
        analyser.trendlineNorm()
        analyser.plotTempNormAngle()

        #print("Half normalised --------------------------")
        #analyser.trendlineHalfNorm()
        #analyser.plotTempHalfNormAngle()
