import pandas as pd

import data
import model

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

x_train, y_train = data.loadTrain()
x_test, id_test = data.loadTest()

CV=6

rf_params = {
	"max_features": 'sqrt'
}

rf_grid_params = {
	"n_estimators": [9,18,27,36,45,54,63],
 	"max_depth": [1, 5, 10, 15, 20, 25, 30],
 	"min_samples_leaf": [1, 2, 4, 6, 8, 10]
}


models = []
#models.append(("logreg",LogisticRegressionCV))
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
	estimators.append(("l",e))
	p = grid.predict(x_test)
	sub = pd.DataFrame({'PassengerId': id_test, 'Survived': p})
	sub.to_csv("output/grid-" + l + ".csv", index=False)

#print("Running VotingClassifier...")
#models = [(label,clf()) for (label,clf) in models]

vc = VotingClassifier(estimators)
vc.fit(x_train, y_train)
p = vc.predict(x_test)
sub = pd.DataFrame({'PassengerId': id_test, 'Survived': p})
sub.to_csv("output/grid-voting.csv", index=False)
