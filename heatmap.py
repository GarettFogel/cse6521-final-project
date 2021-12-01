import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cwd = os.getcwd()
sns.set()
df = pd.read_excel(cwd + "/Conf_mat.xlsx")
labels = df.iloc[:, 0].tolist()
#print(labels)
df.drop(columns=df.columns[0], 
        axis=1, 
        inplace=True)
arr = df.to_numpy()
#print(arr)
ax = sns.heatmap(df, xticklabels=labels, yticklabels=labels)
plt.show()