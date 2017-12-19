#!/usr/bin/env python

from sklearn.model_selection import GridSearchCV

SEED=0

class Model(object):
	def __init__(self, clf, seed=SEED, params={}, cv=10):
		self.clf = clf(**params)

	def train(self, x, y):
		self.clf.fit(x, y)

	def predict(self, x):
		return self.clf.predict(x)

	def fit(self,x,y):
		return self.clf.fit(x,y)

	def gridsearch(self,X,y):
		grid = GridSearchCV(self.clf, self.grid_params, cv=self.cv)
		gfit = grid.fit(X,y)
		model = gfit.best_estimator_
		score = gfit.best_score_
		return model, score

	def bb(self,X,y,test):
		self.train(X,y)
		return self.predict(test)

