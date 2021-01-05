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


#Svm


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
svm_predicted = svclassifier.predict(X_test)
svc_score_train = round(svclassifier.score(X_train, y_train) * 100, 2)
svc_score_test = round(svclassifier.score(X_test, y_test) * 100, 2)

print('SVM Training Score: \n', svc_score_train)
print('SVM Test Score: \n', svc_score_test)
print('Accuracy: \n', accuracy_score(y_test,svm_predicted))

