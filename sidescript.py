from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

df = pd.read_csv("Toddler Autism dataset July 2018.csv")

le = LabelEncoder()
df2 = le.fit_transform(df)

print(df2.head())