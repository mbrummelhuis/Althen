import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

class blockbaxAnalyser():
    def __init__(self, sb_numbers, plot_from_date, plot_till_date,ref=False):
        self.sb_numbers = sb_numbers
        self.dfs = []
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cornlowerblue', 'lightcoral', 'chocolate', 'slateblue', 'orange', 'greenyellow']
        self.y_means = []
        self.y_sigmas = []
        self.poly_degree = 2

        self.ref=ref

        self.plot_from_date = plot_from_date
        self.plot_till_date = plot_till_date

        self.slope = None
        self.intercept = None

        # Trendline settings (min and max temperature)
        self.tl_min = -20
        self.tl_max =  50

    def loadData(self, ext=False,cut=True,ref=False):
        for i in range(len(self.sb_numbers)):
            temp_path = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'), 
                str(self.sb_numbers[i]), str(self.sb_numbers[i]) + '_blockbax_temp.csv')
            y_path = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'), 
                str(self.sb_numbers[i]), str(self.sb_numbers[i]) + '_blockbax_y.csv')
            if self.ref:
                temp_path = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'), 
                str(self.sb_numbers[i]), str(self.sb_numbers[i]) + '_blockbax_temp_ref.csv')
                y_path = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'), 
                str(self.sb_numbers[i]), str(self.sb_numbers[i]) + '_blockbax_y_ref.csv')

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
            if cut:
                delete_before = pd.Timestamp(self.plot_from_date) # Year, month, date
                delete_after = pd.Timestamp(self.plot_till_date)
                data = data.loc[(data.index > delete_before)]
                data = data.loc[(data.index < delete_after)]
                if data.empty:
                    raise RuntimeError("Dataframe is empty, check the plotting time interval.")

            # Calculate means and stdevs            
            self.y_means.append(data["Y value"].mean())
            self.y_sigmas.append(data["Y value"].std())
            if ext == True:
                # Construct extended dataset
                ext_data = [data["Datetime"], data["Temperature"], data["Y value"], (data["Y value"]-self.y_means[i]), (data["Y value"]-self.y_means[i])/self.y_sigmas[i]]
                ext_headers = ['Datetime', 'Temperature', 'Y value', 'Y value (half norm)', 'Y value (norm)']
            else:
                ext_data = [data["Datetime"], data["Temperature"], data["Y value"]]
                ext_headers = ['Datetime', 'Temperature', 'Y value']
                # Append dataframe to object attribute
            
            self.dfs.append(pd.concat(ext_data, axis=1, keys=ext_headers))

    def loadRefData(self,filename):
        names = ["Datetime", "Angle", "Temperature"]
        self.refdf = pd.read_csv(filename, delimiter=',', header=1,engine='python', names=names)

        self.refdf["Datetime"] = pd.to_datetime(self.refdf["Datetime"], format = "%Y-%m-%d %H:%M:%S")
        self.refdf.sort_index(axis=0, ascending=False)
        self.refdf = self.refdf.set_index("Datetime", drop=False)

        delete_before = pd.Timestamp(self.plot_from_date) # Year, month, date
        delete_after = pd.Timestamp(self.plot_till_date)
        self.refdf = self.refdf.loc[(self.refdf.index > delete_before)]
        self.refdf = self.refdf.loc[(self.refdf.index < delete_after)]

        # Parse data to fit into a method similar to polyfitdouble
    
    def determineOffsets(self,zero_date_start, zero_date_end):
        delete_before = pd.Timestamp(zero_date_start)
        delete_after = pd.Timestamp(zero_date_end)
        
        for i in range(len(self.sb_numbers)):
            zero_df = self.dfs[i].loc[(self.dfs[i].index > delete_before)]
            zero_df = zero_df.loc[(zero_df.index < delete_after)]
            offset = zero_df["Y value"].mean()

            # Subtract the 
            self.dfs[i]["Y value"] -= offset

        # Ref offset
        zero_df = self.refdf.loc[(self.refdf.index > delete_before)]
        zero_df = zero_df.loc[(zero_df.index < delete_after)]
        offset = zero_df["Angle"].mean()
        self.refdf["Angle"] -= offset

    def cleanData(self):
        # keep between
        m20_begin = pd.Timestamp("2022-04-28 17:40:00")
        m20_end = pd.Timestamp("2022-04-29 11:40:00")
        # and
        m10_begin = pd.Timestamp("2022-04-29 12:40:00")
        m10_end = pd.Timestamp("2022-04-30 06:40:00")
        # and
        p0_begin = pd.Timestamp("2022-04-30 07:40:00")
        p0_end = pd.Timestamp("2022-05-01 01:40:00")
        # and
        p10_begin = pd.Timestamp("2022-05-01 02:40:00")
        p10_end = pd.Timestamp("2022-05-01 20:40:00")
        # and
        p20_begin = pd.Timestamp("2022-05-01 21:40:00")
        p20_end = pd.Timestamp("2022-05-02 15:40:00")
        # and
        p30_begin = pd.Timestamp("2022-05-02 16:40:00")
        p30_end = pd.Timestamp("2022-05-03 10:40:00")
        # and
        p40_begin = pd.Timestamp("2022-05-03 11:40:00")
        p40_end = pd.Timestamp("2022-05-04 05:40:00")
        # and
        p50_begin = pd.Timestamp("2022-05-04 06:40:00")
        p50_end = pd.Timestamp("2022-05-05 00:40:00")



        for i in range(len(self.sb_numbers)):
            m20_df = self.dfs[i].loc[(self.dfs[i].index > m20_begin)]
            m20_df = m20_df.loc[(m20_df.index<m20_end)]

            m10_df = self.dfs[i].loc[(self.dfs[i].index > m10_begin)]
            m10_df = m10_df.loc[(m10_df.index<m10_end)]

            p0_df = self.dfs[i].loc[(self.dfs[i].index > p0_begin)]
            p0_df = p0_df.loc[(p0_df.index<p0_end)]

            p10_df = self.dfs[i].loc[(self.dfs[i].index > p10_begin)]
            p10_df = p10_df.loc[(p10_df.index<p10_end)]

            p20_df = self.dfs[i].loc[(self.dfs[i].index > p20_begin)]
            p20_df = p20_df.loc[(p20_df.index<p20_end)]

            p30_df = self.dfs[i].loc[(self.dfs[i].index > p30_begin)]
            p30_df = p30_df.loc[(p30_df.index<p30_end)]

            p40_df = self.dfs[i].loc[(self.dfs[i].index > p40_begin)]
            p40_df = p40_df.loc[(p40_df.index<p40_end)]

            p50_df = self.dfs[i].loc[(self.dfs[i].index > p50_begin)]
            p50_df = p50_df.loc[(p50_df.index<p50_end)]

            self.dfs[i]=pd.concat([m20_df, m10_df, p0_df, p10_df, p20_df, p30_df, p40_df, p50_df],axis=0)
            #self.dfs[i] = pd.concat([m10_df, p0_df, p10_df, p20_df], axis=0)

    def matchRefData(self):
        for i in range(len(self.sb_numbers)):
        # For each row of the self.dfs[i], look for the closest matching datetime in self.refdf
        # Finding closest matching datetime in self.refdf: For each row, calculate time difference (absolute). Select smallest time difference
            ref_column = []
            for index,row in self.dfs[i].iterrows():
                current_datetime = row["Datetime"] # datatype timestamp
                self.refdf["Difference"] = abs(self.refdf["Datetime"] - current_datetime)/np.timedelta64(1, 's')

                min_index = self.refdf["Difference"].idxmin()
                ref_column.append(self.refdf.loc[min_index,"Angle"])
            self.dfs[i]["Reference"] = ref_column

            self.dfs[i]["Delta ref"] = self.dfs[i]["Reference"]-self.dfs[i]["Y value"]
            # Delete rows if delta is larger or smaller than 1.0
            self.dfs[i] = self.dfs[i][self.dfs[i]["Delta ref"]< 1.0]
            self.dfs[i] = self.dfs[i][self.dfs[i]["Delta ref"]>-1.0]

            self.dfs[i] = self.dfs[i].drop(columns=['Delta ref'])

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
        for i in range(len(self.sb_numbers)):
            templist = self.dfs[i]['Temperature'].tolist()
            ylist = self.dfs[i]['Y value'].tolist()

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
    
    def trendlineStdTF_per_y(self,i,y_index, y_value):
        """
        Calculates the slope, intercept and r-squared for a trendline of a set with angles normalised w.r.t. mean and stdev.
        """
        if self.slope == None:
            self.slope = []
        if self.intercept ==None:
            self.intercept = []

        temparray = np.array(self.dfs[i].groupby("Y setting").get_group(y_value)['Temp stdtf'].tolist())
        yarray = np.array(self.dfs[i].groupby("Y setting").get_group(y_value)['Y stdtf'].tolist())

        slope, intercept, r_value, p_value, std_err = linregress(temparray, yarray)

        self.slope.append(slope)
        self.intercept.append(intercept)

        print("----------------------------------------------------------------------------------------------------")
        print("Trendline on standard transformed data of SmartBrick ", self.sb_numbers[i], "at angle ", y_value)
        print("slope: %f, intercept: %f" % (self.slope[y_index], self.intercept[y_index]))
        print("R-squared: %f" % r_value**2)

    def trendlineStdTF_per_y_notf_temp(self,i,y_index, y_value):
        """
        Calculates the slope, intercept and r-squared for a trendline of a set with angles normalised w.r.t. mean and stdev.
        """
        if self.slope == None:
            self.slope = []
        if self.intercept ==None:
            self.intercept = []

        temparray = np.array(self.dfs[i].groupby("Y setting").get_group(y_value)['Temperature'].tolist())
        yarray = np.array(self.dfs[i].groupby("Y setting").get_group(y_value)['Y stdtf'].tolist())

        slope, intercept, r_value, p_value, std_err = linregress(temparray, yarray)

        self.slope.append(slope)
        self.intercept.append(intercept)

        print("----------------------------------------------------------------------------------------------------")
        print("Trendline on standard transformed data of SmartBrick ", self.sb_numbers[i], "at angle ", y_value)
        print("slope: %f, intercept: %f" % (self.slope[y_index], self.intercept[y_index]))
        print("R-squared: %f" % r_value**2)

    def trendlineStdTF(self,i):
        """
        Calculates the slope, intercept and r-squared for a trendline of a set with angles normalised w.r.t. mean and stdev.
        """
        if self.slope == None:
            self.slope = []
        if self.intercept ==None:
            self.intercept = []

        temparray = np.array(self.dfs[i]['Temp stdtf'].tolist())
        yarray = np.array(self.dfs[i]['Y stdtf'].tolist())

        slope, intercept, r_value, p_value, std_err = linregress(temparray, yarray)

        self.slope.append(slope)
        self.intercept.append(intercept)

        print("Trendline on standard transformed data of SmartBrick: ", self.sb_numbers[i])
        print("slope: %f, intercept: %f" % (self.slope[i], self.intercept[i]))
        print("R-squared: %f" % r_value**2)

    def dataPrep(self, ref_filename, offset_date_start,offset_date_end):
        self.loadData()
        self.loadRefData(ref_filename)
        self.determineOffsets(offset_date_start,offset_date_end)
        self.cleanData()
        self.matchRefData()

    def polyFitAll(self):
        ylist = []
        templist = []
        reflist = []
        print("Polynomial fit with regression to data of all Smartbricks")
        print('Serial numbers: ', self.sb_numbers)    

        for i in range(len(self.sb_numbers)):
            poly = PolynomialFeatures(degree = self.poly_degree)        
            ylist = ylist + self.dfs[i]['Y value'].tolist()
            templist = templist + self.dfs[i]['Temperature'].tolist()

            reflist = reflist +((self.dfs[i]['Reference'] - self.dfs[i]['Y value']).tolist())
        indep_vars = np.transpose(np.array([ylist, templist]))
        dep_vars = np.array(reflist)

        indep_vars_ = poly.fit_transform(indep_vars)
        print("Feature names: ", poly.get_feature_names_out())
        self.feature_names = poly.get_feature_names_out()

        # Create regression object and perform regression
        self.model = linear_model.LinearRegression(fit_intercept=True)
        self.model.fit(indep_vars_, dep_vars)
        self.model.coef_[0] = self.model.intercept_
        print("Model coefficients: ", self.model.coef_)

    def plotModelAll(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # FIRST: Plot surface from model
        x0_mesh = np.arange(-12, 12, 0.01) # Measured angle mesh
        x1_mesh = np.arange(-22, 55, 0.1) # Measured temperature mesh
        x0_mesh, x1_mesh = np.meshgrid(x0_mesh, x1_mesh)
        
        y_mesh = self.model.coef_[0]*1 + self.model.coef_[1]*x0_mesh + self.model.coef_[2]*x1_mesh + self.model.coef_[3]*x0_mesh*x0_mesh + self.model.coef_[4]*x0_mesh*x1_mesh + self.model.coef_[5]*x1_mesh*x1_mesh

        ax.plot_surface(x0_mesh, x1_mesh, y_mesh) # find out transparant surface map

        # SECOND: Plot scattered data
        for i in range(len(self.sb_numbers)):
            ax.scatter(self.dfs[i]['Y value'], self.dfs[i]['Temperature'], self.dfs[i]['Reference']-self.dfs[i]['Y value'], color=self.colours[i], label=self.sb_numbers[i])

        # Cleanup
        ax.set_xlabel("Measured angle")
        ax.set_ylabel("Measured temperature")
        ax.set_zlabel("Ref - meas angle")
        ax.legend()
        plt.show()
    
    def validate(self, model=None):
        if model == None:
            model=self.model
        for i in range(len(self.sb_numbers)):

            self.dfs[i]["Compensated Y"] = self.dfs[i]["Y value"] + model.coef_[0] + model.coef_[1]*self.dfs[i]["Y value"] + model.coef_[2]*self.dfs[i]["Temperature"]+ model.coef_[3]*self.dfs[i]["Y value"]**2+ model.coef_[4]*self.dfs[i]["Y value"]*self.dfs[i]["Temperature"]+ model.coef_[5]*self.dfs[i]["Temperature"]**2
            self.dfs[i]["Error"] = self.dfs[i]["Reference"] - self.dfs[i]["Compensated Y"]
            self.dfs[i]["Uncompensated error"] = self.dfs[i]["Reference"]-self.dfs[i]["Y value"]
        
        self.errorStatistics()
    
    def plotErrorTemp(self,i):

        plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Error'], label=str(self.sb_numbers[i])+' compensated')
        plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]["Uncompensated error"], label=str(self.sb_numbers[i])+' uncompensated')
    
        plt.legend()
        plt.title("Temperature vs Error (ref-comp)")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Error')
        plt.show()

    def plotErrorAngle(self,i):

        plt.scatter(self.dfs[i]['Y value'], self.dfs[i]['Error'], label=str(self.sb_numbers[i])+' compensated')
        plt.scatter(self.dfs[i]['Y value'], self.dfs[i]["Uncompensated error"], label=str(self.sb_numbers[i])+' uncompensated')
    
        plt.legend()
        plt.title("Measured Y value vs Error (ref-comp)")
        plt.xlabel('Measured Y value')
        plt.ylabel('Error')
        plt.show()
    
    def beforeAfter(self, i):
        plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Compensated Y'],label='After comp') 
        plt.scatter(self.dfs[i]["Temperature"], self.dfs[i]["Reference"], label="Reference")      
        plt.scatter(self.dfs[i]['Temperature'], self.dfs[i]['Y value'],label='Before comp')   

        plt.grid()
        plt.legend()
        plt.title("Difference between ref angle and compensated/measured angle")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Difference')
        plt.show()

    def boxplotComparison(self):
        p20_begin = pd.Timestamp("2022-05-01 21:40:00")
        p20_end = pd.Timestamp("2022-05-02 15:40:00")
        
        self.comparison_df = pd.DataFrame()
        names = []
        for i in range(len(self.sb_numbers)):
            
            p20_df = self.dfs[i].loc[(self.dfs[i].index > p20_begin)]
            p20_df = p20_df.loc[(p20_df.index<p20_end)]

            p20_df = p20_df.loc[(abs(p20_df["Y value"]-5.0) < 1.0)]
            if len(p20_df)>23:
                p20_df.drop(p20_df.tail(1).index,inplace=True)
            
            self.comparison_df["Error "+str(self.sb_numbers[i])] = p20_df["Uncompensated error"].values
            names.append("Error "+str(self.sb_numbers[i]))
        
        #Reference data
        ref_df = self.dfs[0].loc[(self.dfs[0].index > p20_begin)]
        ref_df = ref_df.loc[(ref_df.index < p20_end)]
        ref_df = ref_df.loc[(abs(ref_df["Reference"]-5.0) < 1.0)]
        if len(ref_df) > 23:
            ref_df.drop(ref_df.tail(len(ref_df)-23).index,inplace=True)
        ref_df["Reference"] = ref_df["Reference"]-ref_df["Reference"].mean()
        self.comparison_df["Reference"] = ref_df["Reference"].values
        names.append("Reference")
        print(self.comparison_df)
        print(names)
        
        boxplot = self.comparison_df.boxplot(column=names)
        plt.ylabel("Absolute uncompensated error in degrees")
        plt.xlabel("Smartbrick S/N")
        plt.title("Uncompensated absolute error distribution at 20C and 5 deg")
        plt.show()
    
    def errorStatistics(self):
        for i in range(len(self.sb_numbers)):
            print(str(self.sb_numbers[i]), "---------------------------------------")
            print("Max absolute uncompensated error", self.dfs[i]["Uncompensated error"].max())
            print("Average uncompensated error: ", self.dfs[i]["Uncompensated error"].mean())
            print("Min absolute uncompensated error", self.dfs[i]["Uncompensated error"].min())
            print("Max absolute compensated error: ", self.dfs[i]["Error"].max())
            print("Average compensated error: ", self.dfs[i]["Error"].mean())
            print("Min absolute compensated error: ", self.dfs[i]["Error"].min())
            print("RMSE uncompensated: ", (self.dfs[i]["Uncompensated error"]**2).mean()**0.5)
            print("RMSE compensated: ",  (self.dfs[i]["Error"]**2).mean()**0.5)

            print("Absolute improvement: ", (self.dfs[i]["Uncompensated error"]**2).mean()**0.5-(self.dfs[i]["Error"]**2).mean()**0.5)
            print("(Negative improvement means deterioration.)")
            print(" ")
    
    def errorBoxplots(self):
        self.uncomp_error_df = pd.DataFrame()
        self.comp_error_df = pd.DataFrame()
        names = []

        for i in range(len(self.sb_numbers)):
            temp_df = self.dfs[i]
            
            if len(temp_df) > 852:
                temp_df.drop(temp_df.tail(len(temp_df)-852).index,inplace=True)
            
            self.uncomp_error_df[str(self.sb_numbers[i])] = temp_df["Uncompensated error"].values
            self.comp_error_df[str(self.sb_numbers[i])] = temp_df["Error"].values

            names.append(str(self.sb_numbers[i]))
        
        boxplot = self.uncomp_error_df.boxplot(column=names)
        plt.ylabel("Uncompensated error (deg)")
        plt.xlabel("Smartbrick S/N")
        plt.title("Uncompensated error distribution (deg)")
        plt.show()

        boxplot = self.comp_error_df.boxplot(column=names)
        plt.ylabel("Compensated error (deg)")
        plt.xlabel("Smartbrick S/N")
        plt.title("Compensated error distribution (deg)")
        plt.show()

    

if __name__=="__main__":
    start_time = time.time()
    #sb_numbers = [141419, 141422, 141425, 141427, 141430, 141431, 141433]
    #sbs = [148040, 148054, 148062, 148068, 148070, 148074, 148077, 148084, 148089, 148094, 148101, 148102]

    plot_from_date = "2022-04-28 18:30:00" # put in date string with format YYYY-MM-DD HH:MM:SS
    plot_till_date = "2022-05-05 01:00:00"

    offset_date_start = "2022-05-02 03:40:00"
    offset_date_end = "2022-05-02 05:35:00"

    # sbs = [148088, 148097, 148098, 148099, 148105, 148106, 148107] All from lab tests
    sbs_val = [148088, 148097, 148098, 148099] # Good ones from lab tests
    # sbs = [148042, 148051, 148070, 148071, 148084]
    # sbs = [148107, 148098, 148097, 148091, 148088, 148105, 148076, 148096] # Second lab test
    sbs = [148098, 148097, 148091, 148088, 148076, 148096] # Second lab test without bad data smartbricks

    ref_filename = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'),"Ref","jewell_ref_april.txt")
    ref_val_filename = os.path.join(os.path.abspath(__file__).removesuffix('blockbaxAnalyser.py'),"Ref","jewell_ref_march.txt")

    analyser = blockbaxAnalyser(sbs, plot_from_date, plot_till_date)
    val_analyser = blockbaxAnalyser(sbs_val, plot_from_date, plot_till_date,ref=True)

    analyser.dataPrep(ref_filename, offset_date_start,offset_date_end)
    #val_analyser.dataPrep(ref_val_filename,offset_date_end= , offset_date_start=) # Find out begin and end of 20c and 0 deg
    print(analyser.dfs)
    #print(val_analyser.dfs)

    analyser.polyFitAll() # Make model of test data set
    #val_analyser.validate(model=analyser.model)

    print("Took: ", time.time()-start_time, "seconds")

    # val_analyser.boxplotComparison()
    # val_analyser.errorBoxplots()
