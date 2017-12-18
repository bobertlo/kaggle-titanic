import pandas as pd

import data
import model

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

x_train, y_train = data.loadTrain()
x_test, id_test = data.loadTest()

models = []
models.append(("logreg",LogisticRegressionCV))
models.append(("rf",RandomForestClassifier))
models.append(("ada",AdaBoostClassifier))
models.append(("gb",GradientBoostingClassifier))
models.append(("et",ExtraTreesClassifier))
models.append(("svc",SVC))

preds = []
for l,mi in models:
	m = model.Model(mi)
	print("Training BB " + l + "...")
	bb = m.bb(x_train, y_train, x_test)
	preds.append(bb)
	sub = pd.DataFrame({'PassengerId': id_test, 'Survived': bb})
	sub.to_csv("output/bb-" + l + ".csv", index=False)

print("Running VotingClassifier...")
models = [(label,clf()) for (label,clf) in models]

vc = VotingClassifier(models)
vc.fit(x_train, y_train)
p = vc.predict(x_test)
sub = pd.DataFrame({'PassengerId': id_test, 'Survived': p})
sub.to_csv("output/bb-voting.csv", index=False)