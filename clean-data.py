#!/usr/bin/env python

import pandas as pd
import data

tr_x = pd.read_csv("input/train.csv")
data.treatData(tr_x)
#tr_y = tr_x['Survived']
#tr_x.drop(['Survived','PassengerId'],axis=1,inplace=True)

te_x = pd.read_csv("input/test.csv")
data.treatData(te_x)
#te_id = te_x['PassengerId']
#te_x.drop('PassengerId',axis=1,inplace=True)

tr_x.to_csv("data/train.csv",index=False)
#tr_y.to_csv("data/train_y.csv",index=False)

#te_id.to_csv("data/test_id.csv",index=False)
te_x.to_csv("data/test.csv",index=False)