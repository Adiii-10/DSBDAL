import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

data = pd.read_csv("tecdiv.csv")

data

print("The first five rows are as follows: ")
data.head()

print("The last five rows are as follows: ")
data.tail()

data.describe()

data.info()

print("The column names of the dataset are as follows: ")
data.columns

data.isnull().sum()

sns.boxplot(y=data['First year:   Sem 1'])

sns.boxplot(y=data['First year:   Sem 2'])

sns.boxplot(y=data["Second year:   Sem 1"])

sns.boxplot(y=data["Second year:   Sem 2"])
