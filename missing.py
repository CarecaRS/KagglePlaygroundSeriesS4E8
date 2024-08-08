###
# SEPARATE FILE JUST TO TEST IMPUTATIONS
#       WITH CLASSIFICATION MODELS
###

# Importing necessary packages
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
pd.options.display.float_format = '{:.4f}'.format
%autoindent

# Defines my workspace
os.chdir('/home/thiago/Documentos/MBA USP/Kaggle/Mushrooms/')

# Database import
#teste = pd.read_parquet('datasets/teste_polido.parquet')
#treino = pd.read_parquet('datasets/treino_polido.parquet')
#teste = pd.read_parquet('datasets/teste_prednans.parquet')
#treino = pd.read_parquet('datasets/treino_prednans.parquet')

# Imputation test#1: estimating the most null features first,
# in decreasing order. This being as follows:
#
# TO DO: ring-type
# DONE ALREADY (teste): gill-spacing, cap-surface, gill-attachment
#
# Other features NaN percentage is insignificant,
# so the mode is imputed

# Setting the target feature for the script and normalizing the
# numeric values
target = 'ring-type'
categories = teste.drop(target, axis=1).columns[
        teste.drop(target, axis=1)
        .dtypes == 'object'
        ].values

numerics = teste.columns[teste.dtypes == 'float64']
norm = 'l1'
treino[numerics] = normalize(treino[numerics], norm=norm)
teste[numerics] = normalize(teste[numerics], norm=norm)

#####
# DATASET WRANGLING FOR THE TRAIN SET
#####
check1 = pd.DataFrame(treino[target].value_counts())
check2 = pd.DataFrame(teste[target].value_counts())
print(pd.concat([check1, check2], axis=1))

poucos_gillsp = ('m', 'n', 'w', 'i', 'r', 'y', 'h', 'l', 'k', 't')
mask_treino = treino[target].isin(poucos_gillsp)
treino.loc[mask_treino, target] = 'few_obs'
mask_teste = teste[target].isin(poucos_gillsp)
teste.loc[mask_teste, target] = 'few_obs'

mask_t = teste[target].isnull()  # NaN values
mask_f = teste[target].isnull() == False  # non-NaN values

x_train_temp = teste[mask_f].copy()
y_train_temp = teste[mask_t].copy()

y_train = y_train_temp.drop(target, axis=1).fillna('nan')
x_train = x_train_temp.drop(target, axis=1).fillna('nan')
y_train[target] = y_train_temp[target]
x_train[target] = x_train_temp[target]

# Train/test split
y_train_idx = y_train.index.values
x_train_idx = x_train.index.values
y_train_id = y_train['id'].copy()
x_train_id = x_train['id'].copy()
x_train = x_train.drop('id', axis=1)
y_train = y_train.drop('id', axis=1)

tamanho_treino = 0.80
treino_x, teste_x, treino_y, teste_y = train_test_split(x_train.drop(target, axis=1), x_train[target],
                                                        train_size=tamanho_treino,
                                                        random_state=1)

###
# SKLEARN MODEL 0.90178
###
#skl_hgb = HistGradientBoostingClassifier(loss='log_loss',
#                                         learning_rate=0.001,
#                                         max_iter=300,
#                                         max_leaf_nodes=32,
#                                         max_depth=16,
#                                         categorical_features=categories,
#                                         scoring='loss',
#                                         random_state=1,
#                                         verbose=1
#                                         )
#
#skl_hgb.fit(treino_x, treino_y)
#ypred_skl = skl_hgb.predict(teste_x)
#score_skl = accuracy_score(teste_y, ypred_skl)
#print(f'Accuracy score: {score_skl:.5f}')


###
# CATBOOST CLASSIFIER 0.99518
###
catboost = CatBoostClassifier(cat_features=categories,
                              loss_function='MultiClass',
                              eval_metric='Accuracy',  # testar 'Precision'
                              iterations=500,
                              learning_rate=0.081,
                              random_seed=1,
                              bootstrap_type='Poisson',
                              depth=9,
                              early_stopping_rounds=200,
                              thread_count=12,
                              task_type='GPU',
                              gpu_ram_part=0.95,
                              gpu_cat_features_storage='GpuRam',
                              grow_policy='Depthwise',
                              min_child_samples=39,
                              boosting_type='Plain'
                              )

pool_treino = Pool(treino_x,
                   label=treino_y,
                   cat_features=categories)
pool_teste = Pool(teste_x,
                  label=teste_y,
                  cat_features=categories)

catboost.fit(pool_treino,
             eval_set=(pool_teste),
             verbose=100)
ypred_cat = catboost.predict(teste_x)
score_cat = accuracy_score(teste_y, ypred_cat)
print(f'Accuracy score: {score_cat:.5f}')

# Predicting the NaN values for the target
predicts = catboost.predict(y_train)

to_join = pd.DataFrame(predicts)
to_join['id'] = list(y_train_id)
to_join['idx'] = y_train_idx
to_join = to_join.set_index('idx')

x_train['id'] = x_train_id
y_train['id'] = y_train_id
y_train[target] = to_join[0]

nonans = pd.concat([x_train, y_train], axis=0).sort_values(by='id')
teste[target] = nonans[target]

teste.to_parquet('datasets/teste_prednans.parquet')

for col in teste.columns:
    teste[col] = teste[col].fillna(teste[col].mode()[0])

(teste.isnull().sum()/teste.shape[0])*100
