import pandas as pd

import data
import model

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

x_train, y_train = data.loadTrain()
x_test, id_test = data.loadTest()

models = []
models.append(("logreg",LogisticRegressionCV))
models.append(("rf",RandomForestClassifier))

for l,mi in models:
	m = model.Model(mi)
	print("Training BB " + l)
	bb = m.bb(x_train, y_train, x_test)
	sub = pd.DataFrame({'PassengerId': id_test, 'Survived': bb})
	sub.to_csv("output/bb-" + l + ".csv", index=False)
