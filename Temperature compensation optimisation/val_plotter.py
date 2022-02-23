import pandas as pd
import os
import matplotlib.pyplot as plt


class ValidationPlotter():
    def __init__(self,names=['SB1', 'SB2', 'SB3'],run=1):
        self.path_to_folder = os.path.abspath(__file__).removesuffix(
            'val_plotter.py') + 'Scrubbed data'
        self.dfs = []
        self.names = names
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.filenames = []
        self.coefs = [1.43032321, -0.0427527, -0.0431293802, -0.000369161511, -0.0000074174763, -0.00030177516]
        #self.coefs = [-0.556785599, -0.0921241749, -0.0208732928, -0.000161297557, 0.00102393294, 0.0000534597289] 
        #self.coefs = [-2.58724976, 0.0130776252, 0.00666721386, 0.000330729299, -0.00124732660, 0.000278639638] 

        for i in range(len(self.names)):
            self.filenames.append(os.path.join(self.path_to_folder, 
                'temperature_compensation_val_' + str(run) + '_' + str(i+1) + 
                '_scrubbed_manual.txt'))

    
    def loadData(self):
        for i in range(len(self.names)):
            print(self.filenames[i])
            self.dfs.append(pd.read_csv(self.filenames[i]))
            self.dfs[i]['datetime'] = pd.to_datetime(self.dfs[i]["datetime"], 
                format = "%Y-%m-%d %H:%M:%S")
            
            # Check and correct TD temperature under 0 (they get overflow)
            for n in range(self.dfs[i].shape[0]):
                if self.dfs[i]['temp'][n] > 2000:
                    self.dfs[i].loc[n,"temp"] = self.dfs[i]['temp'][n] - 65536
                else:
                    pass
            
            self.dfs[i]['temp'] = self.dfs[i]['temp'].div(10)
            self.dfs[i]['ref_angle'] = self.dfs[i]['ref_angle'] * -1
            self.dfs[i]['meas_angle'] = self.dfs[i]['meas_angle'].div(1000)
            self.dfs[i]['comp_angle'] = self.dfs[i]['comp_angle'].div(1000)
        return

    
    def plotTimeTemp(self):
        for i in range(len(self.names)):
            plt.plot(self.dfs[i]['datetime'], self.dfs[i]['temp'], 
                color=self.colours[i], label=names[i])
        
        plt.grid()
        plt.legend()
        plt.title("Temperature against time")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def plotTempDiff(self):
        for i in range(len(self.names)):
            plt.scatter(self.dfs[i]['temp'], self.dfs[i]['difference'], 
                color=self.colours[i], label=self.names[i])
        
        plt.grid()
        plt.legend()
        plt.title("Difference compensated angle and ref angle against temperature")
        plt.xlabel("Temperature")
        plt.ylabel("Difference ")
        plt.show()
    
    def validateData(self):
        for i in range(len(self.names)):
            self.dfs[i]['recalculated angle'] = self.dfs[i]['meas_angle']+ self.coefs[0]+self.coefs[1]*self.dfs[i]["meas_angle"]+self.coefs[2]*self.dfs[i]['temp']+self.coefs[3]*self.dfs[i]["meas_angle"]**2 +self.coefs[4]*self.dfs[i]["meas_angle"]*self.dfs[i]["temp"] +self.coefs[5]*self.dfs[i]["temp"]**2
            plt.scatter(self.dfs[i]['temp'], self.dfs[i]['ref_angle']-self.dfs[i]['recalculated angle'], 
            color=self.colours[i], label=self.names[i])
        
        print(self.dfs[i])
        plt.grid()
        plt.legend()
        plt.title("Difference compensated angle and ref angle against temperature")
        plt.xlabel("Temperature")
        plt.ylabel("Difference ")
        plt.show()
    
    def beforeAfter(self, index):
        for i in range(len(self.names)):
            self.dfs[i]['recalculated angle'] = self.dfs[i]['meas_angle']+ self.coefs[0]+self.coefs[1]*self.dfs[i]["meas_angle"]+self.coefs[2]*self.dfs[i]['temp']+self.coefs[3]*self.dfs[i]["meas_angle"]**2 +self.coefs[4]*self.dfs[i]["meas_angle"]*self.dfs[i]["temp"] +self.coefs[5]*self.dfs[i]["temp"]**2

        plt.scatter(self.dfs[index]['temp'], self.dfs[index]['ref_angle']-self.dfs[index]['recalculated angle'], label='After comp')
        plt.scatter(self.dfs[index]['temp'], self.dfs[index]['ref_angle']-self.dfs[index]['meas_angle'], label='Before comp')

        plt.grid()
        plt.legend()
        plt.title("Difference compensated angle and ref angle against temperature")
        plt.xlabel("Temperature")
        plt.ylabel("Difference ")
        plt.show()

if __name__ == "__main__":
    names = ['SB1', 'SB2', 'SB3']
    run = 2
    plotter = ValidationPlotter(names=names, run=run)    
    plotter.loadData()
    plotter.plotTimeTemp()
    plotter.plotTempDiff()
    plotter.validateData()
    for i in range(len(names)):
        plotter.beforeAfter(i)
