import os
from numpy import arccos, arcsin, pi
import pandas as pd
import matplotlib.pyplot as plt


class proefbelastingPlotter():
    def __init__(self, names, plot_from_date = '2022-01-10 00:00:00'):
        self.names = names
        self.path_to_folder = os.path.abspath(__file__).removesuffix(
            'proefbelasting.py') + 'Original data'
        self.dfs = []
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.plot_from_date = plot_from_date
        self.g = 2**14 # Static number, number of units representing 1g on an axis
        self.coefs = [1.43032321, -0.0427527, -0.0431293802, -0.000369161511, -0.0000074174763, -0.00030177516]
        #self.coefs = [-0.556785599, -0.0921241749, -0.0208732928, -0.000161297557, 0.00102393294, 0.0000534597289] 
        #self.coefs = [-2.58724976, 0.0130776252, 0.00666721386, 0.000330729299, -0.00124732660, 0.000278639638] 

    def loadData(self):
        for i in range(len(self.names)):
            path_to_file = os.path.join(self.path_to_folder, 
                "export-device-" + self.names[i] + "-messages.csv")
            self.dfs.append(pd.read_csv(path_to_file, delimiter = ";"))

            # Timestamp to datetime format for different timestamp formats
            try:
                self.dfs[i]["Timestamp"] = pd.to_datetime(self.dfs[i]["Timestamp"], 
                    format = "%d-%m-%Y %H:%M")
            except:
                self.dfs[i]['Timestamp'] = pd.to_datetime(self.dfs[i]["Timestamp"], 
                    format = "%Y-%m-%d %H:%M:%S")
            finally:            
                 # Sort list in most recent first
                self.dfs[i] = self.dfs[i].sort_index(axis=0 , ascending=False)
                # Set the timestamp as index
                self.dfs[i] = self.dfs[i].set_index('Timestamp') 

            delete_before = pd.Timestamp(self.plot_from_date) # Year, month, date
            self.dfs[i] = self.dfs[i].loc[(self.dfs[i].index > delete_before)]

            # Message decoding; initiating lists
            xlst = []
            zlst = []
            voltlst = []
            templst = []
            retrylst = []
            shockslst = []

            # Decrypt payload data into x, y and z values and temperatures
            for message in self.dfs[i]["Data"]:
                try:
                    xlst.append(self.twos_complement(str(message[0:4]), 16))
                    zlst.append(self.twos_complement(str(message[4:8]), 16))
                    voltlst.append(self.twos_complement(str(message[8:12]), 16))
                    templst.append(self.twos_complement(str(message[12:16]), 16)/10.)
                    retrylst.append(self.twos_complement(str(message[16:18]),8))
                    shockslst.append(self.twos_complement(str(message[18:22]), 16))
                except:
                    pass
            
            self.dfs[i]["X value"] = xlst
            self.dfs[i]["Z value"] = zlst
            self.dfs[i]["Voltage"] = voltlst
            self.dfs[i]["Temperature"] = templst
            self.dfs[i]["Retries"] = retrylst
            self.dfs[i]["Shocks"] = shockslst
    
    def processData(self):
        """
        Processes loaded raw data in the dataframes, calculates angle using cos/sin (not tan because we don't have data for all three axes), producing measured angle.
        Measured angle and measured temp are then taken through the compensation formula and the result is added to the dataframe.
        """
        for i in range(len(self.names)):
            self.dfs[i]['Alpha']=arcsin(self.dfs[i]['X value']/self.g)*180/pi
            self.dfs[i]['Gamma']=arccos(self.dfs[i]['Z value']/self.g)*180/pi

            self.dfs[i]['Comp alpha'] = (self.dfs[i]['Alpha'] + 
                                        self.coefs[0] + 
                                        self.coefs[1]*self.dfs[i]["Alpha"] + 
                                        self.coefs[2]*self.dfs[i]['Temperature'] + 
                                        self.coefs[3]*self.dfs[i]["Alpha"]**2 +
                                        self.coefs[4]*self.dfs[i]["Alpha"]*self.dfs[i]["Temperature"] +
                                        self.coefs[5]*self.dfs[i]["Temperature"]**2)

            self.dfs[i]['Comp gamma'] = (self.dfs[i]['Gamma'] + 
                                        self.coefs[0] + 
                                        self.coefs[1]*self.dfs[i]["Gamma"] + 
                                        self.coefs[2]*self.dfs[i]['Temperature'] + 
                                        self.coefs[3]*self.dfs[i]["Gamma"]**2 +
                                        self.coefs[4]*self.dfs[i]["Gamma"]*self.dfs[i]["Temperature"] +
                                        self.coefs[5]*self.dfs[i]["Temperature"]**2)

    def twos_complement(self, hexstr,bits):
        value = int(hexstr,16)
        if value & (1 << (bits-1)):
            value -= 1 << bits
        return value
    
    def plotRawData(self):
        for i in range(len(self.names)):
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6,ncols=1,
                sharex=True)
            fig.set_size_inches(19.2,10.8)
            fig.canvas.manager.set_window_title(self.names[i])

            ax1.plot(self.dfs[i].iloc[ : , [3]], marker = '', linestyle='-')
            ax2.plot(self.dfs[i].iloc[ : , [4]], marker = '', linestyle='-')
            ax3.plot(self.dfs[i].iloc[ : , [5]], marker = '', linestyle='-')
            ax4.plot(self.dfs[i].iloc[ : , [6]], marker = '', linestyle='-')
            ax5.plot(self.dfs[i].iloc[ : , [7]], marker = '', linestyle='-')
            ax6.plot(self.dfs[i].iloc[ : , [8]], marker = '', linestyle='-')

            ax1.set_ylabel('Raw X')
            ax2.set_ylabel('Raw Z')
            ax3.set_ylabel('Voltage [mV]')
            ax4.set_ylabel('Temp [C]')
            ax5.set_ylabel('Retries')
            ax6.set_ylabel('Shocks')

            ax1.grid(True)
            ax2.grid(True)
            ax3.grid(True)
            ax4.grid(True)
            ax5.grid(True)
            ax6.grid(True)

            save_path = os.path.join(os.path.abspath(__file__).removesuffix('proefbelasting.py'),
                'Plots\\')

            plt.tight_layout()
            plt.savefig(save_path + self.names[i] + '.png', dpi=100)
            #plt.show()
    
    def plotProcessedData(self):
        for i in range(len(self.names)):
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1,
                sharex=True)
            fig.set_size_inches(19.2,10.8)
            fig.canvas.manager.set_window_title(self.names[i])

            ax1.plot(self.dfs[i].iloc[ : , [9]], color = self.colours[0], marker = '', linestyle='-', label='Alpha')
            ax1.plot(self.dfs[i].iloc[ : , [11]], color = self.colours[1], marker = '', linestyle='-', label='Comp alpha')
            ax2.plot(self.dfs[i].iloc[ : , [10]], color = self.colours[0], marker = '', linestyle='-', label='Gamma')
            ax2.plot(self.dfs[i].iloc[ : , [12]], color = self.colours[1], marker = '', linestyle='-', label='Comp gamma')
            ax3.plot(self.dfs[i].iloc[ : , [6]], marker = '', linestyle='-', label='Temperature')

            ax1.set_ylabel('Alpha (deg)')
            ax2.set_ylabel('Gamma (deg)')
            ax3.set_ylabel('Temperature (C)')

            ax1.grid(True)
            ax2.grid(True)
            ax3.grid(True)

            save_path = os.path.join(os.path.abspath(__file__).removesuffix('proefbelasting.py'),
                'Plots\\')

            ax1.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig(save_path + self.names[i] + 'processed.png', dpi=100)
            #plt.show()

if __name__ == "__main__":
    names = ['CEA276', 'CEA278', 'CEA27E', 'CEA297', 'CEA298', 'CEA299', 'CEA29A', 'CEA2A1']
    plot_from_date = "2021-08-01 01:00:00" # put in date string with format YYYY-MM-DD HH:MM:SS
    plotter = proefbelastingPlotter(names=names,plot_from_date=plot_from_date)
    plotter.loadData()
    plotter.processData()
    #plotter.plotRawData()
    plotter.plotProcessedData()