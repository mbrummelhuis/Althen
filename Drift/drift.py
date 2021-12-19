from matplotlib import pyplot as plt
import os
import pandas as pd
import math
import numpy as np

class DriftDFCreator:
    def __init__(self, filenames):
        self.path=os.path.abspath(__file__).removesuffix('drift.py')
        self.filenames=filenames

    def plotData(self):

        sb1_list = self.createDF(self.filenames[0])['Calc angle (deg)']
        sb2_list = self.createDF(self.filenames[1])['Calc angle (deg)']
        sb3_list = self.createDF(self.filenames[2])['Calc angle (deg)']
        reflist = self.createDF(self.filenames[0])['Jewell angle']

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4,ncols=1,sharex = True)
        fig.set_size_inches(19.2,10.8)

        ax1.plot(sb1_list, marker = '', linestyle='-')
        ax2.plot(sb2_list, marker = '', linestyle='-')
        ax3.plot(sb3_list, marker = '', linestyle='-')
        ax4.plot(reflist, marker = '', linestyle='-')

        ax1.set_ylabel('SB1 [deg]')
        ax2.set_ylabel('SB2 [deg]')
        ax3.set_ylabel('SB3 [deg]')
        ax4.set_ylabel('Jewell')

        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def createDF(self, filename):
        filename = self.path + filename
        header = ['Timestamp','X', 'Y', 'Z', 'Calc angle (rad)','TD temp','Jewell angle','Jewell temp']

        # Load txt log to dataframe
        full_df = pd.read_csv(filename, header=None, names=header)
        full_df['Timestamp'] = pd.to_datetime(full_df["Timestamp"], format = "%Y-%m-%d %H:%M:%S")

        # Calculate device orientation in deg and add column to df
        deg_angles = []
        rad_angles = []
        for n in range(full_df.shape[0]):
            if full_df["X"][n] < 0:
                rad_angles.append(full_df['Z'][n]/(math.sqrt(full_df['X'][n] ** 2 + full_df['Y'][n] ** 2))*-1)
            elif full_df["X"][n] >= 0:
                rad_angles.append(full_df['Z'][n]/(math.sqrt(full_df['X'][n] ** 2 + full_df['Y'][n] ** 2)))

            deg_angles.append(rad_angles[n]/ math.pi * 180)

        full_df['Calc angle (deg)'] = deg_angles
        full_df['Calc angle (radv2)'] = rad_angles
        # Check and correct TD temperature under 0 (they get overflow)
        for n in range(full_df.shape[0]):
            if full_df['TD temp'][n] > 1000:
                full_df['TD temp'][n] = full_df['TD temp'][n] - 65536
            else:
                pass
        # Divide temps by 10 to get deg Celcius
        full_df['TD temp'] = full_df['TD temp'].div(10)  
        print(full_df)
        return full_df


if __name__ == "__main__":
    plotter = DriftDFCreator(filenames=['SB1_driftlog.txt', 'SB2_driftlog.txt', 'SB3_driftlog.txt'])
    plotter.plotData()