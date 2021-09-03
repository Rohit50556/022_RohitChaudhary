
#Importing libraries
import numpy as np 
import pandas as pd 
import io
import matplotlib.pyplot as plt

# reading the csv file, del 2 columns from the file, checking first few rows of the file
from google.colab import files
uploaded = files.upload()

data = pd.read_csv(io.BytesIO(uploaded['BuyComputer.csv']))

data.drop(columns=['User ID',],axis=1,inplace=True)
data.head()

#Declare label as last column in the source file
label=data['Purchased']
label

#Declaring X as all columns excluding last
X=data[['Age','EstimatedSalary']]
X
