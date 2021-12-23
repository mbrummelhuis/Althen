import pandas as pd
import math
import numpy as np
import os
import matplotlib.pyplot as plt

class Comparator:
    def __init__(self):
        self.path_to_folder = os.path.abspath(__file__).removesuffix('comparator.py') + 'Scrubbed data'
        self.number_smartbricks = 1

        self.df1 = None
        self.df2 = None
        self.df3 = None
    
    def loadData(self):
        """
        Loads the Smartbrick data of the three Smartbricks into three separate dataframes (attributes of class).
        Dataframes are automatically scrubbed and only contain timestamp, measured angle and temperature and ref angle and temperature.
        """            
        # DATAFRAME 1-------------------------------
        filename = self.path_to_folder + "\SB1_tempreplog_scrubbed_manual.txt"
        names = ["Timestamp", "X", "Y", "Z", "del1 c ang", "meas temp", "del2 rep", "ref angle", "ref temp","del3 optemp"]

        # Load data into self.df and calculate angle from accelero data
        self.df1 = pd.read_csv(filename, header=None, names = names)
        self.df1['Timestamp'] = pd.to_datetime(self.df1["Timestamp"], format = "%Y-%m-%d %H:%M:%S")
        meas_ang = []
        
        for n in range(self.df1.shape[0]):
            if self.df1["X"][n] < 0:
                meas_ang.append(self.df1["Z"][n]/math.sqrt(self.df1["X"][n] ** 2 + self.df1["Y"][n] ** 2) / math.pi * 180*-1)
            elif self.df1["X"][n] >= 0:
                meas_ang.append(self.df1["Z"][n]/math.sqrt(self.df1["X"][n] ** 2 + self.df1["Y"][n] ** 2) / math.pi * 180)
        self.df1["meas angle"] = meas_ang

        # Delete unnecessary columns and order 
        self.df1 = self.df1.drop(["X", "Y", "Z", "del1 c ang", "del2 rep", "del3 optemp"], axis=1)

        # Check and correct TD temperature under 0 (they get overflow)
        for n in range(self.df1.shape[0]):
            if self.df1['meas temp'][n] > 1000:
                self.df1['meas temp'][n] = self.df1['meas temp'][n] - 65536
            else:
                pass
        self.df1['meas temp'] = self.df1['meas temp'].div(10) 


        # DATAFRAME 2-------------------------------
        filename = self.path_to_folder + "\SB2_tempreplog_scrubbed_manual.txt"
        names = ["Timestamp", "X", "Y", "Z", "del1 c ang", "meas temp", "del2 rep", "ref angle", "ref temp","del3 optemp"]

        # Load data into self.df and calculate angle from accelero data
        self.df2 = pd.read_csv(filename, header=None, names = names)
        self.df2['Timestamp'] = pd.to_datetime(self.df2["Timestamp"], format = "%Y-%m-%d %H:%M:%S")
        meas_ang = []
        
        for n in range(self.df2.shape[0]):
            if self.df2["X"][n] < 0:
                meas_ang.append(self.df2["Z"][n]/math.sqrt(self.df2["X"][n] ** 2 + self.df2["Y"][n] ** 2) / math.pi * 180*-1)
            elif self.df2["X"][n] >= 0:
                meas_ang.append(self.df2["Z"][n]/math.sqrt(self.df2["X"][n] ** 2 + self.df2["Y"][n] ** 2) / math.pi * 180)
        self.df2["meas angle"] = meas_ang

        # Delete unnecessary columns and order 
        self.df2 = self.df2.drop(["X", "Y", "Z", "del1 c ang", "del2 rep", "del3 optemp"], axis=1)

        # Check and correct TD temperature under 0 (they get overflow)
        for n in range(self.df2.shape[0]):
            if self.df2['meas temp'][n] > 1000:
                self.df2['meas temp'][n] = self.df2['meas temp'][n] - 65536
            else:
                pass
        self.df2['meas temp'] = self.df2['meas temp'].div(10) 

        # DATAFRAME 3-------------------------------
        filename = self.path_to_folder + "\SB3_tempreplog_scrubbed_manual.txt"
        names = ["Timestamp", "X", "Y", "Z", "del1 c ang", "meas temp", "del2 rep", "ref angle", "ref temp","del3 optemp"]

        # Load data into self.df and calculate angle from accelero data
        self.df3 = pd.read_csv(filename, header=None, names = names)
        self.df3['Timestamp'] = pd.to_datetime(self.df3["Timestamp"], format = "%Y-%m-%d %H:%M:%S")
        meas_ang = []
        
        for n in range(self.df3.shape[0]):
            if self.df3["X"][n] < 0:
                meas_ang.append(self.df3["Z"][n]/math.sqrt(self.df3["X"][n] ** 2 + self.df3["Y"][n] ** 2) / math.pi * 180*-1)
            elif self.df3["X"][n] >= 0:
                meas_ang.append(self.df3["Z"][n]/math.sqrt(self.df3["X"][n] ** 2 + self.df3["Y"][n] ** 2) / math.pi * 180)
        self.df3["meas angle"] = meas_ang

        # Delete unnecessary columns and order 
        self.df3 = self.df3.drop(["X", "Y", "Z", "del1 c ang", "del2 rep", "del3 optemp"], axis=1)

        # Check and correct TD temperature under 0 (they get overflow)
        for n in range(self.df3.shape[0]):
            if self.df3['meas temp'][n] > 1000:
                self.df3['meas temp'][n] = self.df3['meas temp'][n] - 65536
            else:
                pass
        self.df3['meas temp'] = self.df3['meas temp'].div(10) 

    def plotTempData(self):
        """
        Plots scatterplots of the data, displaying the different Smartbricks in different colours.
        """
        plt.scatter(self.df1['ref temp'], self.df1['meas temp']-self.df1['ref temp'], color='b', label="SB1")
        plt.scatter(self.df2['ref temp'], self.df2['meas temp']-self.df2['ref temp'], color='r', label="SB2")
        plt.scatter(self.df3['ref temp'], self.df3['meas temp']-self.df3['ref temp'], color='g', label="SB3")
        plt.grid()
        plt.legend()
        plt.xlabel('Reference temperature')
        plt.ylabel('Measured temperature')
        plt.show()

    def plotAngleData(self):
        """
        Plots scatterplots of the angle data, displaying the differnet Smartbricks in different colours.
        """
        plt.scatter(self.df1['ref angle'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['ref angle'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['ref angle'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.legend()
        plt.xlabel('Reference angle')
        plt.ylabel('Measured angle')
        plt.show()

if __name__ == "__main__":
    comparator = Comparator()
    comparator.loadData()
    #comparator.plotTempData()
    comparator.plotAngleData()

