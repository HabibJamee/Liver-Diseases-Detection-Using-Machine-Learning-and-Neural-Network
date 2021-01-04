#import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder

#path derectory
liver_df=pd.read_csv('')

#data Preprocessing
liver_df['Albumin_and_Globulin_Ratio'].mean()
liver_df=liver_df.fillna(0.94)



sns.countplot(data=liver_df, x = 'Dataset', label='Count')

LD, NLD = liver_df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)



sns.countplot(data=liver_df, x = 'Gender', label='Count')

M, F = liver_df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)




sns.catplot(x="Age", y="Gender", hue="Dataset", data=liver_df);
sns.catplot(x="Dataset", y="Albumin", data=liver_df);



X = liver_df.drop(['Gender','Dataset'], axis=1)
X.head(3)



#split the dataset
from sklearn.model_selection import train_test_split
X=liver_df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=liver_df['Dataset']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
