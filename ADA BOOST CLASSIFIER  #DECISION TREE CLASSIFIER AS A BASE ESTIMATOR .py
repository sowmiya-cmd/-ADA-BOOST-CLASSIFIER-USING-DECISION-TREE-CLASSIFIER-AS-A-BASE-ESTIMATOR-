#ADA BOOST CLASSIFIER USING #DECISION TREE CLASSIFIER AS A BASE ESTIMATOR 

from sklearn.ensemble import AdaBoostClassifier
# initialize the base classifier 
base_cls = DecisionTreeClassifier() 
from sklearn import metrics
# Create adaboost classifer object

abc = AdaBoostClassifier(n_estimators=50,base_estimator = base_cls,
                         learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)
score=metrics.accuracy_score(y_test, y_pred)
score =score*100
print("Accuracy Using Adaboost:",score)

#Cohen's kappa score with ADABOOST classifier and with decision tree as a base estimator 
from sklearn.metrics import cohen_kappa_score
abc.fit(X_train,y_train)
abc.score(X_test,y_test)
Y_pred = abc.predict(X_test)
cohen_score = cohen_kappa_score(y_test, Y_pred)
print("Cohen's Kappa index ||Decision Tree with Adaboost :" ,cohen_score)
from sklearn.metrics import mean_squared_error
print("RMSE using AdaBoost:",mean_squared_error(y_test, y_pred))



