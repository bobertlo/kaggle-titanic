import pandas as pd

import data
import model

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

x_train, y_train = data.loadTrain()
x_test, id_test = data.loadTest()

CV=6

rf_params = {
	"max_features": 'sqrt'
}

rf_grid_params = {
	"n_estimators": [2,3,4,5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17, 18,27,36,45,54,63],
 	"max_depth": [1, 5, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30],
 	"min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 10]
}

knn_params = { }

knn_grid_params = {
	"n_neighbors": [5,6,7,8,9,10,11,13,15,17,19,20,21,23,25,30,35,40,45,50,60],
	"metric": ['euclidean','manhattan'],
	"weights": ['uniform','distance'],
}

ada_params = { }

ada_grid_params = {
	"n_estimators": [1, 2, 5, 10, 20, 30, 35, 40, 45, 47, 50, 52, 53, 54, 55, 57, 60, 70],
	"learning_rate": [ 0.001, 0.003, 0.01, 0.03, 0.1, 0.15, 0.75, 0.2, 0.225, 0.25, 0.3, 0.4, 0.7, 1, 1.5],
	"algorithm": ['SAMME', 'SAMME.R'],
}

et_params = {}

et_grid_params = {
	"max_features": [1,2,3,4,5,6,7,10,11,12,'auto'],
	"min_samples_split": [2,3,7,10,11,12,13,14,15,20,25],
	"min_samples_leaf": [1,2,3,4,5,6,8,9,10,11,12,15,20],
	"n_estimators": [1,2,3,4,5,6,7,10,20,25,30,35,40,50,60],
	"criterion": ['gini','entropy'],
}

gb_params = { }

gb_grid_params = {
	"learning_rate": [0.001, 0.003, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.07, 0.1, 0.3, 0.5],
	"max_depth": [2,3,4,5,6,8,9,10,11,12],
	"min_samples_leaf": [1, 3, 7, 10, 12, 13, 14, 15, 16, 17, 20, 30, 90],
	"max_features": ['sqrt','log2',None],
}

svc_params = { "probability": True }
svc_grid_params = {
	"gamma": [0.001, 0.001, 0.003, 0.007, 0.01, 0.15, 0.02, 0.25, 0.3, 0.1, 1],
	"C": [1, 5, 7, 8, 9, 10, 11, 13, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 100, 300, 1000],
}


models = []
#models.append(("logreg",LogisticRegressionCV))
models.append(("svc",SVC,svc_params,svc_grid_params))
models.append(("gb",GradientBoostingClassifier,gb_params,gb_grid_params))
models.append(("et",ExtraTreesClassifier,et_params,et_grid_params))
models.append(("ada",AdaBoostClassifier,ada_params,ada_grid_params))
models.append(("knn",KNeighborsClassifier,knn_params,knn_grid_params))
models.append(("rf",RandomForestClassifier,rf_params,rf_grid_params))
#models.append(("ada",AdaBoostClassifier))
#models.append(("gb",GradientBoostingClassifier))
#models.append(("et",ExtraTreesClassifier))
#models.append(("svc",SVC))

estimators = []
for l,mi,pi,pgi in models:
	m = mi(**pi)
	print("Runing GridSearch on " + l + "...")
	grid = GridSearchCV(estimator=m, param_grid=pgi, cv=CV, verbose=1, n_jobs=-1)
	grid.fit(x_train, y_train)
	print(grid.best_params_)
	print(grid.best_score_)
	e = grid.best_estimator_
	estimators.append((l,e))
	p = grid.predict(x_test)
	sub = pd.DataFrame({'PassengerId': id_test, 'Survived': p})
	sub.to_csv("output/grid-" + l + ".csv", index=False)

#print("Running VotingClassifier...")
#models = [(label,clf()) for (label,clf) in models]

vc = VotingClassifier(estimators, voting='hard', weights=[1,1,1,1,1,3])
vc.fit(x_train, y_train)
print(vc.score(x_train,y_train))
p = vc.predict(x_test)
sub = pd.DataFrame({'PassengerId': id_test, 'Survived': p})
sub.to_csv("output/grid-voting.csv", index=False)
