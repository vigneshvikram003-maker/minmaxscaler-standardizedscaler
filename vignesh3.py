import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
import pandas as pd
tips=sns.load_dataset("tips")
print(tips)
print("Min Max Scaler")
numeric_col=tips.select_dtypes(include='number').columns
scaler=MinMaxScaler()
tips_normalized=pd.DataFrame(scaler.fit_transform(tips[numeric_col]),columns=numeric_col)
print(tips_normalized.head())
print("Standard Scaler")

numeric_col1=tips.select_dtypes(include="number").columns
scaler=StandardScaler()
tips_standardized=pd.DataFrame(scaler.fit_transform(tips[numeric_col1]),columns=numeric_col1)
print(tips_standardized.head())

plt.figure(figsize=(8,6))
plt.hist(tips['total_bill'], bins=10, edgecolor='black')
plt.title("Histogram of Total Bill")
plt.xlabel("Total Bill")
plt.ylabel("Frequency")
plt.show()
