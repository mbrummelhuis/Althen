import pandas as pd
import math
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

class Comparator:
    def __init__(self, names=['SB1', 'SB2', 'SB3']):
        self.path_to_folder = os.path.abspath(__file__).removesuffix('comparator.py') + 'Scrubbed data'
        self.names = names
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.dfs = []
        self.filenames = []
        self.offsets = [0.128912, -0.5807, -1.09694]
        for n in range(len(self.names)):
            self.filenames.append(os.path.join(self.path_to_folder, self.names[n] + '_tempreplog_scrubbed_manual.txt'))

        self.df1 = None
        self.df2 = None
        self.df3 = None
    
    def loadDatav2(self):
        for i in range(len(self.names)):
            # Load data into self.df and calculate angle from accelero data
            names = ["Timestamp", "X", "Y", "Z", "del1 c ang", "meas temp", "del2 rep", "ref angle", "ref temp","del3 optemp"]
            self.dfs.append(pd.read_csv(self.filenames[i], header=None, names=names))
            self.dfs[i]['Timestamp'] = pd.to_datetime(self.dfs[i]["Timestamp"], format = "%Y-%m-%d %H:%M:%S")

            meas_ang = []
            for n in range(self.dfs[i].shape[0]):
                if self.dfs[i]["Z"][n] < 0:
                    meas_ang.append(abs(self.dfs[i]["Z"][n]/math.sqrt(self.dfs[i]["X"][n] ** 2 + self.dfs[i]["Y"][n] ** 2) / 
                        math.pi * 180))
                elif self.dfs[i]["Z"][n] >= 0:
                    meas_ang.append(abs(self.dfs[i]["Z"][n]/math.sqrt(self.dfs[i]["X"][n] ** 2 + self.dfs[i]["Y"][n] ** 2) / 
                        math.pi * 180)*-1)
            
            self.dfs[i]["meas angle"] = meas_ang
            self.dfs[i]["meas angle"] = self.dfs[i]["meas angle"]-self.offsets[i]

            # Delete unnecessary columns and order 
            self.dfs[i] = self.dfs[i].drop(["del1 c ang", "del2 rep", "del3 optemp"], axis=1)

            # Check and correct TD temperature under 0 (they get overflow)
            for n in range(self.dfs[i].shape[0]):
                if self.dfs[i]['meas temp'][n] > 2000:
                    self.dfs[i].loc[n,"meas temp"] = self.dfs[i]['meas temp'][n] - 65536
                else:
                    pass
            
            self.dfs[i]['meas temp']= self.dfs[i]["meas temp"]/10. 

            # Reorder columns
            cols = self.dfs[i].columns.tolist()
            cols = cols[:4] + [cols[5]] + [cols[7]] + [cols[6]] + [cols[4]]
            self.dfs[i] = self.dfs[i][cols]
        return

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
            if self.df1["Z"][n] < 0:
                meas_ang.append(abs(self.df1["Z"][n]/math.sqrt(self.df1["X"][n] ** 2 + self.df1["Y"][n] ** 2) / math.pi * 180))
            elif self.df1["Z"][n] >= 0:
                meas_ang.append(abs(self.df1["Z"][n]/math.sqrt(self.df1["X"][n] ** 2 + self.df1["Y"][n] ** 2) / math.pi * 180)*-1)
        self.df1["meas angle"] = meas_ang

        # Delete unnecessary columns and order 
        self.df1 = self.df1.drop(["del1 c ang", "del2 rep", "del3 optemp"], axis=1)

        # Check and correct TD temperature under 0 (they get overflow)
        for n in range(self.df1.shape[0]):
            if self.df1['meas temp'][n] > 2000:
                self.df1['meas temp'][n] = self.df1['meas temp'][n] - 65536
            else:
                pass
        self.df1['meas temp'] = self.df1['meas temp'].div(10) 

        # Reorder columns
        cols1 = self.df1.columns.tolist()
        cols1 = cols1[:4] + [cols1[5]] + [cols1[7]] + [cols1[6]] + [cols1[4]]
        self.df1 = self.df1[cols1]

        # DATAFRAME 2-------------------------------
        filename = self.path_to_folder + "\SB2_tempreplog_scrubbed_manual.txt"
        names = ["Timestamp", "X", "Y", "Z", "del1 c ang", "meas temp", "del2 rep", "ref angle", "ref temp","del3 optemp"]

        # Load data into self.df and calculate angle from accelero data
        self.df2 = pd.read_csv(filename, header=None, names = names)
        self.df2['Timestamp'] = pd.to_datetime(self.df2["Timestamp"], format = "%Y-%m-%d %H:%M:%S")
        meas_ang = []
        
        for n in range(self.df2.shape[0]):
            if self.df2["Z"][n] < 0:
                meas_ang.append(abs(self.df2["Z"][n]/math.sqrt(self.df2["X"][n] ** 2 + self.df2["Y"][n] ** 2) / math.pi * 180))
            elif self.df2["Z"][n] >= 0:
                meas_ang.append(abs(self.df2["Z"][n]/math.sqrt(self.df2["X"][n] ** 2 + self.df2["Y"][n] ** 2) / math.pi * 180)*-1)
        self.df2["meas angle"] = meas_ang

        # Delete unnecessary columns and order 
        self.df2 = self.df2.drop(["del1 c ang", "del2 rep", "del3 optemp"], axis=1)

        # Check and correct TD temperature under 0 (they get overflow)
        for n in range(self.df2.shape[0]):
            if self.df2['meas temp'][n] > 2000:
                self.df2['meas temp'][n] = self.df2['meas temp'][n] - 65536
            else:
                pass
        self.df2['meas temp'] = self.df2['meas temp'].div(10) 

        # Reorder columns
        cols2 = self.df2.columns.tolist()
        cols2 = cols2[:4] + [cols2[5]] + [cols2[7]] + [cols2[6]] + [cols2[4]]
        self.df2 = self.df2[cols2]

        # DATAFRAME 3-------------------------------
        filename = self.path_to_folder + "\SB3_tempreplog_scrubbed_manual.txt"
        names = ["Timestamp", "X", "Y", "Z", "del1 c ang", "meas temp", "del2 rep", "ref angle", "ref temp","del3 optemp"]

        # Load data into self.df and calculate angle from accelero data
        self.df3 = pd.read_csv(filename, header=None, names = names)
        self.df3['Timestamp'] = pd.to_datetime(self.df3["Timestamp"], format = "%Y-%m-%d %H:%M:%S")
        meas_ang = []
        
        for n in range(self.df3.shape[0]):
            if self.df3["Z"][n] < 0:
                meas_ang.append(abs(self.df3["Z"][n]/math.sqrt(self.df3["X"][n] ** 2 + self.df3["Y"][n] ** 2) / math.pi * 180))
            elif self.df3["Z"][n] >= 0:
                meas_ang.append(abs(self.df3["Z"][n]/math.sqrt(self.df3["X"][n] ** 2 + self.df3["Y"][n] ** 2) / math.pi * 180)*-1)
        self.df3["meas angle"] = meas_ang

        # Delete unnecessary columns and order 
        self.df3 = self.df3.drop(["del1 c ang", "del2 rep", "del3 optemp"], axis=1)

        # Check and correct TD temperature under 0 (they get overflow)
        for n in range(self.df3.shape[0]):
            if self.df3['meas temp'][n] > 2000:
                self.df3['meas temp'][n] = self.df3['meas temp'][n] - 65536
            else:
                pass
        self.df3['meas temp'] = self.df3['meas temp'].div(10)

        # Reorder columns
        cols3 = self.df3.columns.tolist()
        cols3 = cols3[:4] + [cols3[5]] + [cols3[7]] + [cols3[6]] + [cols3[4]]
        self.df3 = self.df3[cols3] 
        return

    def plotTempData(self):
        """
        Plots scatterplots of the data, displaying the different Smartbricks in different colours.
        """
        self.loadDatav2()
        for i in range(len(self.names)):
            plt.scatter(self.dfs[i]['ref temp'], self.dfs[i]['meas temp']-self.dfs[i]['ref temp'], 
                color=self.colours[i], label="SB"+str(i+1))
        plt.grid()
        plt.legend()
        plt.title("Temperature dependence of temperature sensor")
        plt.xlabel('Reference temperature')
        plt.ylabel('Tmeas - Tref')
        plt.show()
        return

    def plotArefAdiff(self):
        """
        Plots scatterplots of the angle data, displaying the differnet Smartbricks in different colours.
        """
        self.loadDatav2()
        for i in range(len(self.names)):
            plt.scatter(self.dfs[i]['ref angle'], self.dfs[i]['meas angle']-self.dfs[i]['ref angle'],
                color = self.colours[i], label="SB"+str(i+1))

        plt.grid()
        plt.legend()
        plt.title("Temperature dependence of angle sensor")
        plt.xlabel('Reference angle')
        plt.ylabel('Ameas - Aref')
        plt.show()
        return
    
    def plotTmeasAdiff(self):
        """
        Plots scatterplot of angle difference data vs temperature, displaying the differnet Smartbricks in different colours.
        """
        self.loadDatav2()
        for i in range(len(self.names)):
            plt.scatter(self.dfs[i]['meas temp'], self.dfs[i]['meas angle']-self.dfs[i]['ref angle'], 
                color=self.colours[i], label="SB"+str(i+1))

        plt.grid()
        plt.legend()
        plt.title("Temperature dependence of angle sensor")
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show()
        return

    def plotAmeasAdiff(self):
        self.loadDatav2()
        for i in range(len(self.names)):
            plt.scatter(self.dfs[i]['meas angle'], self.dfs[i]['meas angle']-self.dfs[i]['ref angle'],
                color=self.colours[i], label="SB"+str(i+1))

        plt.grid()
        plt.legend()
        plt.title("Temperature dependence of angle sensor")
        plt.xlabel('Measured angle')
        plt.ylabel('Ameas - Aref')
        plt.show() 
        return       
    
    def plotTmeasAmeasAdiff(self):
        """
        Plots 3D scatterplot with dependent variable (Ameas-Aref) vs measured temperature (Tmeas) and measured angle (Ameas).
        Different Smartbricks are plotted in different colours.
        """
        self.loadDatav2()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in range(len(self.names)):
            ax.scatter(self.dfs[i]['meas temp'], self.dfs[i]['meas angle'], 
                self.dfs[i]['meas angle']-self.dfs[i]['ref angle'], color=self.colours[i], label="SB"+str(i+1))

        ax.grid()
        ax.legend()

        ax.set_xlabel('Measured temperature')
        ax.set_ylabel('Measured angle')
        ax.set_zlabel('Ameas - Aref')
        plt.show()
        return
    
    def plotTmeasAmeasAref(self):
        self.loadDatav2()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in range(len(self.names)):
            ax.scatter(self.dfs[i]['meas temp'], self.dfs[i]['meas angle'], 
                self.dfs[i]['ref angle'], color=self.colours[i], label="SB"+str(i+1))

        ax.grid()
        ax.legend()

        ax.set_xlabel('Measured temperature')
        ax.set_ylabel('Measured angle')
        ax.set_zlabel('Reference angle')
        plt.show()  
        return

    def plotTmeasAmeasAfactor1(self):
        """
        Plots 3D scatterplot with dependent variable (Ameas/Aref) vs measured temperature (Tmeas) and measured angle (Ameas).
        Different Smartbricks are plotted in different colours.
        """
        self.loadDatav2()
        # Delete -1, 0, 1 angles
        droplist = []
        dfs_dropped = []
        for i in range(len(self.names)):
            droplist.append([])
            for n in range(self.dfs[i].shape[0]):
                if abs(self.dfs[i]['ref angle'][n]) < 1.1:
                    droplist[i].append(n)
            
            dfs_dropped.append(self.dfs[i].drop(droplist[i], axis=0, inplace=True))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(len(self.names)):
            ax.scatter(self.dfs[i]['meas temp'], self.dfs[i]['meas angle'], self.dfs[i]['meas angle'] / self.dfs[i]['ref angle'], 
                color=self.colours[i], label="SB"+str(i+1))

        ax.grid()
        ax.legend()

        ax.set_xlabel('Measured temperature')
        ax.set_ylabel('Measured angle')
        ax.set_zlabel('Ameas / Aref')
        plt.show()
        return

    def plotTmeasAmeasAfactor2(self):
        """
        Plots 3D scatterplot with dependent variable (Ameas/Aref) vs measured temperature (Tmeas) and measured angle (Ameas).
        Different Smartbricks are plotted in different colours.
        """
        self.loadData()
        # Delete -1, 0, 1 angles
        droplist1 = []
        droplist2 = []
        droplist3 = []

        for n in range(self.df1.shape[0]):
            if abs(self.df1['meas angle'][n]) < 5:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if abs(self.df2['meas angle'][n]) < 5:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if abs(self.df3['meas angle'][n]) < 5:
                droplist3.append(n)

        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(self.df1['meas temp'], self.df1['meas angle'], self.df1['ref angle'] / self.df1['meas angle'], 
            color='b', label="SB1")
        ax.scatter(self.df2['meas temp'], self.df2['meas angle'], self.df2['ref angle'] / self.df2['meas angle'], 
            color='r', label="SB2")
        ax.scatter(self.df3['meas temp'], self.df3['meas angle'], self.df3['ref angle'] / self.df3['meas angle'], 
            color='g', label="SB3")
        ax.grid()
        ax.legend()

        ax.set_xlabel('Measured temperature')
        ax.set_ylabel('Measured angle')
        ax.set_zlabel('Ameas / Aref')
        plt.show()
        return

    def plotNeg10(self):
        self.loadData()

        # Delete other angles
        droplist1 = []
        droplist2 = []
        droplist3 = []
        for n in range(self.df1.shape[0]):
            if self.df1['ref angle'][n] > -9:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if self.df2['ref angle'][n] > -9:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if self.df3['ref angle'][n] > -9:
                droplist3.append(n)
        
        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        plt.scatter(self.df1['meas temp'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['meas temp'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['meas temp'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.title('Negative 10')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show() 
        return         

    def plotNeg2(self):
        self.loadData()

        # Delete other angles
        droplist1 = []
        droplist2 = []
        droplist3 = []
        for n in range(self.df1.shape[0]):
            if self.df1['ref angle'][n] > -1.5 or self.df1['ref angle'][n] < -2.5:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if self.df2['ref angle'][n] > -1.5 or self.df2['ref angle'][n] < -2.5:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if self.df3['ref angle'][n] > -1.5 or self.df2['ref angle'][n] < -2.5:
                droplist3.append(n)
        
        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        plt.scatter(self.df1['meas temp'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['meas temp'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['meas temp'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.title('Negative 2')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show() 
        return 
    
    def plotNeg1(self):
        self.loadData()

        # Delete other angles
        droplist1 = []
        droplist2 = []
        droplist3 = []
        for n in range(self.df1.shape[0]):
            if self.df1['ref angle'][n] > -0.5 or self.df1['ref angle'][n] < -1.5:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if self.df2['ref angle'][n] > -0.5 or self.df2['ref angle'][n] < -1.5:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if self.df3['ref angle'][n] > -0.5 or self.df2['ref angle'][n] < -1.5:
                droplist3.append(n)
        
        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        plt.scatter(self.df1['meas temp'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['meas temp'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['meas temp'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.title('Negative 1')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show()
        return

    def plot0(self):
        self.loadData()

        # Delete other angles
        droplist1 = []
        droplist2 = []
        droplist3 = []
        for n in range(self.df1.shape[0]):
            if self.df1['ref angle'][n] < -0.5 or self.df1['ref angle'][n] > 0.5:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if self.df2['ref angle'][n] < -0.5 or self.df2['ref angle'][n] > 0.5:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if self.df3['ref angle'][n] < -0.5 or self.df2['ref angle'][n] > 0.5:
                droplist3.append(n)
        
        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        plt.scatter(self.df1['meas temp'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['meas temp'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['meas temp'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.title('Zero')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show()
        return

    def plotPos1(self):
        self.loadData()

        # Delete other angles
        droplist1 = []
        droplist2 = []
        droplist3 = []
        for n in range(self.df1.shape[0]):
            if self.df1['ref angle'][n] < 0.5 or self.df1['ref angle'][n] > 1.5:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if self.df2['ref angle'][n] < 0.5 or self.df2['ref angle'][n] > 1.5:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if self.df3['ref angle'][n] < 0.5 or self.df2['ref angle'][n] > 1.5:
                droplist3.append(n)
        
        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        plt.scatter(self.df1['meas temp'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['meas temp'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['meas temp'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.title('Positive 1')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show()
        return

    def plotPos2(self):
        self.loadData()

        # Delete other angles
        droplist1 = []
        droplist2 = []
        droplist3 = []
        for n in range(self.df1.shape[0]):
            if self.df1['ref angle'][n] < 1.5 or self.df1['ref angle'][n] > 2.5:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if self.df2['ref angle'][n] < 1.5 or self.df2['ref angle'][n] > 2.5:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if self.df3['ref angle'][n] < 1.5 or self.df2['ref angle'][n] > 2.5:
                droplist3.append(n)
        
        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        plt.scatter(self.df1['meas temp'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['meas temp'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['meas temp'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.title('Positive 2')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show()
        return

    def plotPos10(self):
        self.loadData()

        # Delete other angles
        droplist1 = []
        droplist2 = []
        droplist3 = []
        for n in range(self.df1.shape[0]):
            if self.df1['ref angle'][n] < 9:
                droplist1.append(n)

        for n in range(self.df2.shape[0]):
            if self.df2['ref angle'][n] < 9:
                droplist2.append(n)

        for n in range(self.df3.shape[0]):
            if self.df3['ref angle'][n] < 9:
                droplist3.append(n)
        
        self.df1.drop(droplist1, axis=0, inplace=True)
        self.df2.drop(droplist2, axis=0, inplace=True)
        self.df3.drop(droplist3, axis=0, inplace=True)

        plt.scatter(self.df1['meas temp'], self.df1['meas angle']-self.df1['ref angle'], color='b', label="SB1")
        plt.scatter(self.df2['meas temp'], self.df2['meas angle']-self.df2['ref angle'], color='r', label="SB2")
        plt.scatter(self.df3['meas temp'], self.df3['meas angle']-self.df3['ref angle'], color='g', label="SB3")
        plt.grid()
        plt.title('Positive 10')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Ameas - Aref')
        plt.show()
        return

    def plotZdata(self):
        """
        Plots Z data for all smartbricks against temperature. First plots all data in one plot, then plots data per angle setting.
        """
        self.loadDatav2()
        for i in range(len(self.names)):
            plt.scatter(self.dfs[i]['meas temp'], self.dfs[i]['Z'], color = self.colours[i], label='SB'+str(i+1))
        plt.grid()
        plt.title('Raw Z against temperature')
        plt.legend()
        plt.xlabel('Measured temperature')
        plt.ylabel('Raw Z')
        plt.show()
        return

    def printFullDF1(self):
        print("Dataframe Smartbrick 1 -------------------------------------------------------------")
        print(self.df1.to_string())

    def printFullDF2(self):
        print("Dataframe Smartbrick 2 -------------------------------------------------------------")
        print(self.df2.to_string())

    def printFullDF3(self):
        print("Dataframe Smartbrick 3 -------------------------------------------------------------")
        print(self.df3.to_string())

if __name__ == "__main__":
    names = ['SB1', 'SB2', 'SB3']
    comparator = Comparator(names=names)
    comparator.loadDatav2()
    #comparator.plotTempData()
    #comparator.plotArefAdiff()
    #comparator.plotTmeasAdiff()
    #comparator.plotAmeasAdiff()
    #comparator.plotTmeasAmeasAdiff()
    #comparator.plotTmeasAmeasAref()
    #comparator.plotTmeasAmeasAfactor1()

    #comparator.plotZdata()
