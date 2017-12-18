#!/usr/bin/env python

import pandas as pd
import numpy as np

def treatData(d):
	d['Sex'].replace(['male','female'],[0,1],inplace=True)

	d['Embarked'].fillna('S',inplace=True)
	d['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

	d['Title'] = d.Name.str.extract('([A-Za-z]+)\.')
	d['Title'].replace(['Mlle','Ms'],'Miss', inplace=True)
	d['Title'].replace('Mme','Mrs',inplace=True)
	d['Title'].replace(['Capt','Col','Don','Major','Rev','Jonkheer','Sir'],'Rare',inplace=True)
	d['Title'].replace(['Countess','Lady','Dona'],'Rare',inplace=True)
	d['Title'].replace(['Dr'],'Rare',inplace=True)

	d['HasAge']=1
	d.loc[d.Age.isnull(),'HasAge']=0

	d['HasCabin']=1
	d.loc[d.Cabin.isnull(),'HasCabin']=0

	d.loc[d.Age.isnull()&(d.Title=='Mr'),'Age']=33
	d.loc[d.Age.isnull()&(d.Title=='Mrs'),'Age']=36
	d.loc[d.Age.isnull()&(d.Title=='Master'),'Age']=5
	d.loc[d.Age.isnull()&(d.Title=='Miss'),'Age']=22
	d.loc[d.Age.isnull()&(d.Title=='Rare'),'Age']=46

	# split ages into bins
	d['AgeBin'] = pd.cut(d['Age'],[0,16,32,48,65,1000],labels=[0,1,2,3,4])

	# family size is SibSp + Parch + 1
	d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

	# family size of 1 mean a person is alone
	d['Alone'] = 0
	d.loc[d['FamilySize'] == 1, 'Alone'] = 1

	d['Fare'] = d['Fare'].fillna(d['Fare'].median())
	d['Fare'] = d['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

	# drop unused features
	d.drop(['Age','Name','Cabin','Ticket', 'SibSp'],axis=1,inplace=True)
