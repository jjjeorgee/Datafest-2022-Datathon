# Datafest 2022 Datathon
 <img width="682" alt="Screenshot 2022-10-04 105922" src="https://user-images.githubusercontent.com/98137996/193791779-3cf76594-b46b-4b1a-be49-f142719c4e65.png">
 
 ## Libraries
 ```
 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import lightgbm
from lightgbm import LGBMClassifier

 ```
 
 ## Datasets
 [Kaggle ML Datathon Datasets](https://www.kaggle.com/competitions/datafestafrica-ml-hackathon/data)

## Abstract
The aim of this project is to train a Machine Learning model to help a Financial Industry predict who is likely to complete an E-process application.

## Data Wrangling and EDA
[Wrangling and EDA Notebook](https://nbviewer.org/github/jjjeorgee/Datafest-2022-Datathon/blob/Main/DatafestAfrica%20ML%20Hackathon.ipynb)

## Machine Learning Model
### Using cross validation with LighthGBM
```
df_train_dummy['kfold'] = -1
kf = KFold(n_splits = 6, shuffle = True, random_state = 50)
for fold, (train_indices, valid_indices) in enumerate(kf.split(X = df_train_dummy)):
    df_train_dummy.loc[valid_indices, 'kfold'] = fold
useful_features = df_train_dummy.columns.difference(['Entry_id', 'risk_score_4', 'risk_score_5',
                                                     'e_signed', 'total_months_employed','kfold'])
test = df_test_dummy[df_test_dummy.columns.difference(['Entry_id','risk_score_4', 'risk_score_5',
                                                       'e_signed', 'total_months_employed', 'kfold'])]
predictions = []
lgb_score = []
for fold in range(6):
    x_train = df_train_dummy[df_train_dummy['kfold'] != fold].reset_index(drop = True)
    x_valid = df_train_dummy[df_train_dummy['kfold'] == fold].reset_index(drop = True)
    
    y_train = x_train['e_signed']
    y_valid = x_valid['e_signed']
    
    x_train = x_train[useful_features]
    x_valid = x_valid[useful_features]

    
    model = LGBMClassifier(random_state = 42, n_jobs = -1, n_estimators = 1474, learning_rate = 0.03372831097120445,
                           colsample_bytree = 0.7257259813712209, max_depth = 6)
    model.fit(x_train, y_train, early_stopping_rounds = 300, eval_set = [(x_valid, y_valid)], verbose = 1000)
    y_pred = model.predict_proba(x_valid)
    test_pred = model.predict_proba(test)
    predictions.append(test_pred[:, -1])
    score = roc_auc_score( y_valid, y_pred[:,-1])
    lgb_score.append(score)
    print(fold, score)
print(np.mean(lgb_score))
0 0.7086579515760892
1 0.6971319318720783
2 0.6928242225592235
3 0.7170228623150022
4 0.7069782802968488
5 0.7075509259259259
0.7050276957575279
final_prediction = np.mean(np.column_stack(predictions), axis = 1)

final_prediction = np.where(final_prediction >= 0.5, 1,0)

Entry_id = df_test[['Entry_id']]
e_signed = pd.DataFrame({'e_signed' : final_prediction})

lgbm_cv_sub = pd.concat([Entry_id, e_signed], axis = 1)
lgbm_cv_sub.to_csv('lgbm_CV_sub.csv', index = False)

```

## Conclusions
After carrying out this exploratory data analysis, the following conclusions were made:

1. Most applicants fall within the age brackets of 41-50(3,575/28.58%) and 31-40(3,499/27.97%)
2. 1979 of all applicants who signed electronically are in the 31-40 age group, making it the highest occurring age group.
3. 6,766/54.06% of all applicants signed electronically.
4. Applicants requesting for higher amounts (>$1500) are less likely to sign electronically.
5. Applicants with higher incomes are less likely to sign electronically.
6. 4,059 (60%) of all applicants who signed electronically were not homeowners.

## Written in
> Python

## Authors
> Zion Oluwasegun 👤

> Neto Anyama 👤

> Oladimeji Olaniyan 👤
