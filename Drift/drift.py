from matplotlib import pyplot as plt
import os
import pandas as pd
import math

path = os.path.abspath(__file__).removesuffix('drift.py')
filename = path + 'SB1_driftlog.txt'

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
