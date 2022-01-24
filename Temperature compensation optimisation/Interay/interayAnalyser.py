import os
import pandas as pd
import matplotlib.pyplot as plt

class interayAnalyser:
    def __init__(self, sb_numbers):
        self.sb_numbers = sb_numbers
        self.filenames = []
        self.paths = []
        self.dfs = []
        for index in range(len(self.sb_numbers)):
            self.filenames.append('SB_' + str(self.sb_numbers[index]) + '_Interay_alldata.xlsx')
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
                'SmartBrick ('+str(self.sb_numbers[i])+') Y':'Y',
                'SmartBrick ('+str(self.sb_numbers[i])+') Y ADC':'Y ADC'}, axis=1)
            self.dfs[i]['Time'] = pd.to_datetime(self.dfs[i]["Time"], format = "%Y-%m-%d %H:%M:%S")
            print(self.dfs[i])
        return
    
    def plotTempX(self):
        """
        Creates a scatterplot of the different smartbrick's measured temperature against X angle
        """
        colours = ['b', 'r', 'g']
        for i in range(len(self.dfs)):
            plt.scatter(self.dfs[i]['Temp (C)'], self.dfs[i]['X'], color=colours[i], label=str(self.sb_numbers[i]))

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
        colours = ['b', 'r', 'g']
        for i in range(len(self.dfs)):
            plt.scatter(self.dfs[i]['Temp (C)'], self.dfs[i]['Y'], color=colours[i], label=str(self.sb_numbers[i]))

        plt.grid()
        plt.legend()
        plt.title("Temperature dependence of angle Y")
        plt.xlabel('Measured Temperature')
        plt.ylabel('Measured Angle Y')
        plt.show()
    
    

if __name__ == '__main__':
    SB_numbers = [141421, 141442]
    analyser = interayAnalyser(sb_numbers=SB_numbers)
    analyser.loadData()
    analyser.plotTempY()
    analyser.plotTempX()



