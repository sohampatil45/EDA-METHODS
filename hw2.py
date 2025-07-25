import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

#load data
df=pd.read_csv('product_inventory_dataset.csv')

#detect missing values
print(df.head())
print(df.info())
print(df.isnull())

#handle missing values
df['UnitsInStock'].fillna(96.0,inplace=True)
df['SupplierRating'].fillna(5.1,inplace=True)

#data visulization
sns.barplot(x='Category',y='UnitsInStock',data=df)
msno.heatmap(df)
plt.show()

#outliers
Q1=df['ReorderLevel'].quantile(0.25)
print(Q1)
Q3=df['SupplierRating'].quantile(0.75)
print(Q3)
IQR=Q3-Q1
print(IQR)
lower_bound=Q1-1.5*IQR
print(lower_bound)
upper_bound=Q3+1.5*IQR
print(upper_bound)

#feature engineering
df['UnitPrice1']=np.log(df['UnitPrice']+1)
df['TotalCost']=df['UnitsInStock']+df['UnitPrice']

#group anyalisis
grouped_data=df.groupby('Category')['UnitPrice'].sum()
print(grouped_data)

print(df)