#Import Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
rf_predicted = random_forest.predict(X_test)
random_forest_score_train = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)

print('Random Forest Training Score: \n', random_forest_score_train)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))


#Import Libraries
from sklearn.svm import SVC
#SVM
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
svm_predicted = svclassifier.predict(X_test)
svc_score_train = round(svclassifier.score(X_train, y_train) * 100, 2)
svc_score_test = round(svclassifier.score(X_test, y_test) * 100, 2)

print('SVM Training Score: \n', svc_score_train)
print('SVM Test Score: \n', svc_score_test)
print('Accuracy: \n', accuracy_score(y_test,svm_predicted))


 
#Import Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)

