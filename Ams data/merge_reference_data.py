import os
import pandas as pd

dir = os.path.dirname(__file__)
folder = os.path.join(dir, "ground_truth")
# init empty dataframe
df = pd.DataFrame()

for filename in os.listdir(folder):
    f = os.path.join(folder, filename)
    section = filename[0:2].replace(' ', '')
    data = pd.read_excel(f, engine='openpyxl')
    data['location'] = f"Overamstel sectie {section}"
    df = pd.concat([df, data], ignore_index = True)

df.to_csv(os.path.join(dir, "ref_measurements.csv"))
print(df.head())