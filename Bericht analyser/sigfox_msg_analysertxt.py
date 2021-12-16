import math
import pandas as pd

# Function from internet to convert hex strings into signed integers of specified bits
def twos_complement(hexstr,bits):
    value = int(hexstr,16)
    if value & (1 << (bits-1)):
        value -= 1 << bits
    return value

try:
    df = pd.read_csv("sigfox_messages.txt", delimiter = "\t")
    df = df.iloc[: , :-4]
    source = "txt"
except:
    print("ERROR: File not found.")

xlst = []
ylst = []
zlst = []
tlst = []

for message in df["Data / Decoding"]:
    xlst.append(twos_complement(str(message[0:4]), 16))
    ylst.append(twos_complement(str(message[4:8]), 16))
    zlst.append(twos_complement(str(message[8:12]), 16))
    tlst.append(twos_complement(str(message[12:20]), 32)/10.)

df["X value"] = xlst
df["Y value"] = ylst
df["Z value"] = zlst
df["Temperature"] = tlst

print(df)
 
    