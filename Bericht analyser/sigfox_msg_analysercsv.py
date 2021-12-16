import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import datetime

# Function from internet to convert hex strings into signed integers of specified number of bits
def twos_complement(hexstr,bits):
    value = int(hexstr,16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value

asking = True
while asking == True:
    # Ask for device ID and transform to filename
    devID = input("Enter device ID: ")
    filename = "export-device-" + devID + "-messages.csv"

    # Open CSV file to dataframe
    try:
        df = pd.read_csv(filename , delimiter = ";")
        asking = False
    except:
        print("ERROR: File not found. Try again.")

# Convert timestamp to datetime format, backend was overhauled during registration of different devices, so datetime format may be different hence the 'try'statement
try:
    df['Timestamp'] = pd.to_datetime(df["Timestamp"], format = "%d-%m-%Y %H:%M")
except:
    df['Timestamp'] = pd.to_datetime(df["Timestamp"], format = "%Y-%m-%d %H:%M:%S")
finally:            
    df = df.sort_index(axis=0 , ascending=False) # Sort list in most recent first
    df = df.set_index('Timestamp') # Set the timestamp as index
print(df)
# Delete data before set date
delete_before = pd.Timestamp('2021-12-12 00:00:00') # Year, month, date
df = df.loc[(df.index > delete_before)]
print(df)

xlst = []
zlst = []
voltlst = []
templst = []
retrylst = []
shockslst = []

# Decrypt payload data into x, y and z values and temperatures
for message in df["Data"]:
    try:
        xlst.append(twos_complement(str(message[0:4]), 16))
        zlst.append(twos_complement(str(message[4:8]), 16))
        voltlst.append(twos_complement(str(message[8:12]), 16))
        templst.append(twos_complement(str(message[12:16]), 16)/10.)
        retrylst.append(twos_complement(str(message[16:18]),8))
        shockslst.append(twos_complement(str(message[18:22]), 16))

    except:
        print("Message format does not match expected format.")
    finally:
        pass

df["X value"] = xlst
df["Z value"] = zlst
df["Voltage"] = voltlst
df["Temperature"] = templst
df["Retries"] = retrylst
df["Shocks"] = shockslst

print(df)

# Plot figure

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6,ncols=1,sharex=True)
fig.set_size_inches(19.2,10.8)
fig.canvas.set_window_title(devID)

ax1.plot(df.iloc[ : , [3]], marker = '', linestyle='-')
ax2.plot(df.iloc[ : , [4]], marker = '', linestyle='-')
ax3.plot(df.iloc[ : , [5]], marker = '', linestyle='-')
ax4.plot(df.iloc[ : , [6]], marker = '', linestyle='-')
ax5.plot(df.iloc[ : , [7]], marker = '', linestyle='-')
ax6.plot(df.iloc[ : , [8]], marker = '', linestyle='-')

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

plt.tight_layout()
plt.savefig(devID + '.png', dpi=100)
plt.show()