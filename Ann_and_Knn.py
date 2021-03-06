from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

dataset=pd.read_csv('')
dataset.isnull().sum()
dataset['Albumin_and_Globulin_Ratio'].mean()
dataset=dataset.fillna(0.94)
dataset.head()

X = dataset.drop(['Gender','Dataset'], axis=1)
X.head(3)

X=dataset[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=dataset['Dataset']

#Import Libraries
from sklearn.model_selection import train_test_split

#Train test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

#ANN

model=Sequential()
model.add(Dense(12,input_dim=9,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=10)
scores1=model.evaluate(X_train,y_train)
print("Training Accuracy")
print(model.metrics_names[1],scores1[1]*100)


scores2=model.predict(X_test)

print("Testing Accuracy")
print(model.metrics_names[1],scores2[1]*100)

#KNN
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=10,p=1)
KNN.fit(X_train, y_train)
knn_predicted=KNN.predict(X_test)
knn_score_train = round(KNN.score(X_train, y_train) * 100, 2)
knn_score_test = round(KNN.score(X_test, y_test) * 100, 2)
print('KNN Training Score: \n', knn_score_train)
print('KNN Test Score: \n', knn_score_test)
