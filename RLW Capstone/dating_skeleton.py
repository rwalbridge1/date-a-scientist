import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv('profiles.csv')
#print(df.head())
#print(df.sign.value_counts())
print(df.sex.value_counts())
print()
print(df.income.value_counts())
print()
print(df.height.value_counts())
print()
print(df.pets.value_counts())

"""
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()
"""
