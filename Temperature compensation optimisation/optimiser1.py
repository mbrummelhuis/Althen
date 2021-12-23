import pandas as pd
import math
import numpy as np
import os
import matplotlib.pyplot as plt

def polynomial(x, coeffs):
    """ 
    Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    y_vals = []
    for n in range(len(x)):
        y = 0
        p = x[n]
        for i in range(o):
            y += coeffs[i]*p**i
        y_vals.append(y)
    return y_vals

path = os.path.abspath(__file__).removesuffix('optimiser1.py')
filename = path + 'SB1_tempreplog_scrubbed.txt'

header = ['Timestamp', 'X', 'Y', 'Z', 'Calc angle (rad)', 'TD temp', 'Repetitions', 'Jewell angle', 'Jewell temp', 'Opsens temp']

# Load txt log to dataframe
full_df = pd.read_csv(filename, header=None, names=header)

# Convert timestamp to datetime format
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

# Create a new barebones df with only the information necessary for the regression
# Calc angle (deg), Jewell angle, TD temp

data = [full_df["Calc angle (deg)"], full_df["Jewell angle"], full_df["TD temp"]]

headers = ["a_meas", "a_true", "temp"]

df = pd.concat(data, axis=1, keys=headers)

diff =[]
ratio = []
for n in range(df.shape[0]):
    diff.append(df["a_true"][n] - df["a_meas"][n])
    ratio.append(df["a_true"][n] / df["a_meas"][n])

df["diff"] = diff
df["ratio"] = ratio

# Plot the data for analysis
x = df["temp"]
y = df["a_true"]
plt.scatter(df["temp"], df["a_meas"], label = 'Measured')
plt.scatter(df["temp"], df["a_true"], label = 'True')
plt.xlabel('temp')
plt.ylabel('angle')
plt.legend()
plt.grid()
plt.show()


# Comment one y out
x = df['temp'].tolist()
y = df["diff"].tolist()
#y = df['ratio'].tolist()

coeffs = np.polyfit(x,y,1).tolist()
coeffs.reverse()
print(coeffs)

plt.scatter(x,y)
x.sort()
plt.plot(x,polynomial(x,coeffs))
plt.xlabel('temp')
plt.ylabel('diff')
plt.show()

# Try for dif
# Try for ratio
# Try for higher order polynomials

