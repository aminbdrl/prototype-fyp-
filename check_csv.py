import pandas as pd

df = pd.read_csv("data/kelantan_extended.csv")

print(df.columns.tolist())
print(df.head())
