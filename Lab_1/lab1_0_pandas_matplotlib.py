# -*- coding: utf-8 -*-
"""Lab1_0_pandas_matplotlib.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Af0bNtIs5OC9xk9-97T9YmIwX_nraZdG
"""

from google.colab import drive
drive.mount("/content/drive")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/content/drive/MyDrive/ML/Lab_1/Data_for_Transformation.csv')
print(data)

plt.scatter(data["Age"],data["Salary"])
plt.show()

plt.hist(data["Salary"], bins = 10, color = "blue")

plt.bar(data["Country"],data["Salary"],color="green")
plt.xlabel("Country")
plt.ylabel("Salary")