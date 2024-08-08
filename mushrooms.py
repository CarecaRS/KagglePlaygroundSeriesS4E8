###
# WORK NOTES (a.k.a. 'to-do list')
###
# - The parquet files (treino_polido/teste_polido) have only categorical NaN values, all other information is already cleaned/set up.
# - The categorical NaN values should have a proper imputation, maybe from Anacor to estabilish correlations and get it right
# - Try to do an ensemble model with binary values, from sklearn and catboost

# Importing necessary packages
import pandas as pd
import numpy as np
import os
# import tensorflow as tf
# import io
import matplotlib.pyplot as plt
from datetime import datetime
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef  # , fbeta_score, make_scorer
from sklearn.preprocessing import normalize
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier, Pool
# import xgboost as xgb
pd.options.display.float_format = '{:.4f}'.format
%autoindent

# Some functions for recording the models development
def modelo_tf_string(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def realiza_registro_tf():
    registro_geral = pd.read_csv('registros/registros_resultados.csv')
    registro_geral = pd.concat([registro_geral, registro_atual], ignore_index=True)
    registro_geral.to_csv('registros/registros_resultados.csv', index=False)
    with open("./registros/registros_modelagem.txt", "a") as registros:
        registros.write("########## INÍCIO DE REGISTRO - MODELO TensorFlow " + nome_modelo + " ##########\n")
        registros.write("\nInformações geradas em " + datetime.now().strftime("%d-%m-%Y") + " às " + datetime.now().strftime("%H:%M") + ".\n")
        registros.write('Parâmetros do modelo:\n')
        registros.write(modelo_tf_string(modelo))
        registros.write("   --> Score local do modelo (accuracy): " + str(score))
        registros.write("\n\n########## FINAL DE REGISTRO - MODELO TensorFlow " + nome_modelo + " ##########\n\n\n\n")
    print('Novo registro realizado com sucesso!')


def realiza_registro_skl():
    registro_geral = pd.read_csv('registros/registros_resultados.csv')
    registro_geral = pd.concat([registro_geral, registro_atual], ignore_index=True)
    registro_geral.to_csv('registros/registros_resultados.csv', index=False)
    with open("./registros/registros_modelagem.txt", "a") as registros:
        registros.write("########## INÍCIO DE REGISTRO - MODELO Sklearn HistGBC " + nome_modelo + " ##########\n")
        registros.write("\nInformações registradas em " + datetime.now().strftime("%d-%m-%Y") + " às " + datetime.now().strftime("%H:%M") + ".\n\n")
        registros.write('Parâmetros do modelo:\n')
        registros.write(str(skl_hgb.get_params()))
        registros.write(f"\n   Normalização dos valores numéricos: {norm}")
        registros.write("\n\n   --> Score local do modelo (MCC): " + str(round(score_skl, 5)))
        registros.write("\n\n########## FINAL DE REGISTRO - MODELO Sklearn HistGBC " + nome_modelo + " ##########\n\n\n\n")
    print('Novo registro realizado com sucesso!')


def realiza_registro_cat():
    registro_geral = pd.read_csv('registros/registros_resultados.csv')
    registro_geral = pd.concat([registro_geral, registro_atual], ignore_index=True)
    registro_geral.to_csv('registros/registros_resultados.csv', index=False)
    with open("./registros/registros_modelagem.txt", "a") as registros:
        registros.write("########## INÍCIO DE REGISTRO - MODELO CatBoost Classifier " + nome_modelo + " ##########\n")
        registros.write("\nInformações registradas em " + datetime.now().strftime("%d-%m-%Y") + " às " + datetime.now().strftime("%H:%M") + ".\n\n")
        registros.write('Parâmetros do modelo:\n')
        registros.write(str(catboost.get_params()))
        registros.write(f"\n   Normalização dos valores numéricos: {norm}")
        registros.write("\n\n   --> Score local do modelo (MCC): " + str(round(score_cat, 5)))
        registros.write("\n\n########## FINAL DE REGISTRO - MODELO CatBoost Classifier " + nome_modelo + " ##########\n\n\n\n")
    print('Novo registro realizado com sucesso!')


def realiza_registro_xgb():
    registro_geral = pd.read_csv('registros/registros_resultados.csv')
    registro_geral = pd.concat([registro_geral, registro_atual], ignore_index=True)
    registro_geral.to_csv('registros/registros_resultados.csv', index=False)
    with open("./registros/registros_modelagem.txt", "a") as registros:
        registros.write("########## INÍCIO DE REGISTRO - MODELO XGBoost Classifier " + nome_modelo + " ##########\n")
        registros.write("\nInformações registradas em " + datetime.now().strftime("%d-%m-%Y") + " às " + datetime.now().strftime("%H:%M") + ".\n\n")
        registros.write('Parâmetros do modelo:\n')
        registros.write(str(classif_xgb.get_params()))
        registros.write(f"\n   Normalização dos valores numéricos: {norm}")
        registros.write("\n\n   --> Score local do modelo (MCC): " + str(round(score_xgb, 5)))
        registros.write("\n\n########## FINAL DE REGISTRO - MODELO XGBoost Classifier " + nome_modelo + " ##########\n\n\n\n")
    print('Novo registro realizado com sucesso!')


# Defines my workspace
os.chdir('/home/thiago/Documentos/MBA USP/Kaggle/Mushrooms/')

# Database import
#teste_completo = pd.read_parquet('test.parquet')
#treino_completo = pd.read_parquet('train.parquet')
#teste = teste_completo.copy()
#treino = treino_completo.copy()

teste = pd.read_parquet('datasets/teste_prednans.parquet')
treino = pd.read_parquet('datasets/treino_prednans.parquet')

# Checking for features null values over 50%, dropping off said features
mask = (treino.isnull().sum()/treino.shape[0])*100 > 50
fora = treino.columns[mask].values
treino = treino.drop(fora, axis=1)
teste = teste.drop(fora, axis=1)

###
# DATA WRANGLING
###

# The database has some strange values for each feature (words or numbers
# instead of letters, for example). This section cleans it by replacing
# odd records with 'bug' string

obj = treino.columns[treino.dtypes == object]
for i in obj:
    mask = treino[i].str.len() > 1
    treino.loc[mask, i] = 'bug'

obj_teste = teste.columns[teste.dtypes == object]
for i in obj_teste:
    mask = teste[i].str.len() > 1
    teste.loc[mask, i] = 'bug'

# Some features still have odd values not embraced by the loop above
treino.loc[treino[treino['cap-shape'] == '8'].index, 'cap-shape'] = 'bug'
treino.loc[treino[treino['cap-color'] == '7'].index, 'cap-color'] = 'bug'
treino.loc[treino[treino['gill-attachment'] == '1'].index, 'gill-attachment'] = 'bug'
treino.loc[treino[treino['gill-spacing'] == '0'].index, 'gill-spacing'] = 'bug'
treino.loc[treino[treino['gill-spacing'] == '1'].index, 'gill-spacing'] = 'bug'
treino.loc[treino[treino['gill-color'] == '4'].index, 'gill-color'] = 'bug'
treino.loc[treino[treino['gill-color'] == '5'].index, 'gill-color'] = 'bug'
treino.loc[treino[treino['ring-type'] == '4'].index, 'ring-type'] = 'bug'
treino.loc[treino[treino['ring-type'] == '1'].index, 'ring-type'] = 'bug'
treino.loc[treino[treino['ring-type'] == '2'].index, 'ring-type'] = 'bug'
treino.loc[treino[treino['habitat'] == '4'].index, 'habitat'] = 'bug'

teste.loc[teste[teste['cap-shape'] == '8'].index, 'cap-shape'] = 'bug'
teste.loc[teste[teste['cap-shape'] == '6'].index, 'cap-shape'] = 'bug'
teste.loc[teste[teste['gill-spacing'] == '5'].index, 'gill-spacing'] = 'bug'
teste.loc[teste[teste['gill-color'] == '4'].index, 'gill-color'] = 'bug'
teste.loc[teste[teste['ring-type'] == '1'].index, 'ring-type'] = 'bug'
teste.loc[teste[teste['ring-type'] == '2'].index, 'ring-type'] = 'bug'

# Train and test imputations are done separately, to avoid data leakage
# Imputation of numeric NaN values with median
teste['cap-diameter'] = teste['cap-diameter'].fillna(teste['cap-diameter'].median())
teste['stem-height'] = teste['stem-height'].fillna(teste['stem-height'].median())
treino['cap-diameter'] = treino['cap-diameter'].fillna(treino['cap-diameter'].median())

# Imputation of categorical NaN values with string 'nan' to do a quick
# naive model, gotta improve this later
treino_na = treino.fillna('nan')
teste_na = teste.fillna('nan')

# Normalizing the numeric values just before modeling, so I'm able to test
# the results with different normalization hyperparameters
categories = treino.drop('class', axis=1).columns[treino.drop('class', axis=1).dtypes == 'object'].values
numerics = treino.columns[treino.dtypes == 'float64']
norm = 'l1'  # 'l1' or 'l2'

treino_na[numerics] = normalize(treino_na[numerics], norm=norm)
teste_na[numerics] = normalize(teste_na[numerics], norm=norm)

# Train/test split
treino = treino.drop('id', axis=1)
teste = teste.drop('id', axis=1)
tamanho_treino = 0.80
target = 'class'

treino_x, teste_x, treino_y, teste_y = train_test_split(treino.drop(target, axis=1), treino[target],
                                                        train_size=tamanho_treino,
                                                        random_state=1)


###
# SKLEARN MODEL
###
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")
modelo_skl = 'Sklearn HistGBC'
fbeta = make_scorer(fbeta_score, beta=0.66, pos_label='p')
skl_hgb = HistGradientBoostingClassifier(loss='log_loss',
                                         learning_rate=0.101,
                                         max_iter=500,
                                         max_leaf_nodes=32,
                                         max_depth=16,
                                         categorical_features=categories,
                                         scoring=fbeta,
                                         random_state=1,
                                         verbose=1
                                         )

skl_hgb.fit(treino_x, treino_y)
ypred_skl = skl_hgb.predict(teste_x)
score_skl = matthews_corrcoef(teste_y, ypred_skl)
print(f'Score Sklearn MCC local: {score_skl:.5f}')


# Record the informations regarding the model and score in local files
registro_geral = pd.read_csv('registros/registros_resultados.csv')
registro_atual = pd.DataFrame([[pd.to_datetime(nome_modelo), modelo_skl, round(score_skl, 5)]])
registro_atual.columns = ('Dia e Hora', 'Modelo', 'Score (MCC)')
if registro_geral.iloc[registro_geral.shape[0]-1]['Dia e Hora'] == str(pd.to_datetime(nome_modelo)):
    print('Trabalhando com mesmo modelo')
else:
    realiza_registro_skl()

# Estimates the challenge values, creating the file for submission on Kaggle
y_skl_final = skl_hgb.predict(teste_na)
submissao = pd.read_parquet('datasets/submission.parquet')
submissao['class'] = y_skl_final
submissao.to_csv('submissoes/submissao_sklearn_'+nome_modelo+'.csv', index=False)


###
# CATBOOST CLASSIFIER
###

# CatBoost needs the target variable to be binary (1 or 0), so the backup and transformations below
treino_yback = treino_y.copy()
teste_yback = teste_y.copy()
treino_y[treino_y == 'e'] = 1
treino_y[treino_y == 'p'] = 0
teste_y[teste_y == 'e'] = 1
teste_y[teste_y == 'p'] = 0

nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")
modelo_cat = 'CatBoostClassifier'
catboost = CatBoostClassifier(cat_features=categories,
                              loss_function='CrossEntropy',
                              eval_metric='MCC',
                              iterations=4000,
                              learning_rate=0.081,
                              random_seed=1,
                              bootstrap_type='Poisson',
                              bagging_temperature=6,
                              depth=9,
                              early_stopping_rounds=200,
                              thread_count=12,
                              task_type='GPU',
                              target_border=0.05,
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
score_cat = matthews_corrcoef(teste_y.convert_dtypes('numeric'), ypred_cat)
print(f'Score CatBoost MCC local: {score_cat:.5f}')


# Record the informations regarding the model and score in local files
registro_geral = pd.read_csv('registros/registros_resultados.csv')
registro_atual = pd.DataFrame([[pd.to_datetime(nome_modelo), modelo_cat, round(score_cat, 5)]])
registro_atual.columns = ('Dia e Hora', 'Modelo', 'Score (MCC)')
if registro_geral.iloc[registro_geral.shape[0]-1]['Dia e Hora'] == str(pd.to_datetime(nome_modelo)):
    print('Trabalhando com mesmo modelo')
else:
    realiza_registro_cat()

# Estimates the challenge values, creating the file for submission on Kaggle
y_cat_final = catboost.predict(teste)
submissao = pd.read_parquet('datasets/submission.parquet')
submissao['class'] = y_cat_final
mask1 = submissao['class'] == 1
mask0 = submissao['class'] == 0
submissao.loc[mask1, 'class'] = 'e'
submissao.loc[mask0, 'class'] = 'p'
submissao.to_csv('submissoes/submissao_catboost_'+nome_modelo+'.csv', index=False)


###
# XGBOOST CLASSIFIER
###
# XGBoost also needs the target variable to be binary
treino_yback = treino_y.copy()
teste_yback = teste_y.copy()
treino_y[treino_y == 'e'] = 1
treino_y[treino_y == 'p'] = 0
teste_y[teste_y == 'e'] = 1
teste_y[teste_y == 'p'] = 0

treino_x[categories] = treino_x[categories].astype('category')
teste_x[categories] = teste_x[categories].astype('category')

nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")
modelo_xgb = 'XGB Classifier'
classif_xgb = xgb.XGBClassifier(booster="gbtree",
                                tree_method="approx",
                                n_estimators=500,
                                early_stopping_rounds=300,
                                device="cuda",
                                nthread=12,
                                eta=0.011,
                                max_depth=14,
                                max_leaves=135,
                                objective='binary:hinge',
                                eval_metric='error',
                                seed=1,
                                enable_categorical=True
                                )

classif_xgb.fit(treino_x, treino_y,
                eval_set=[(treino_x, treino_y), (teste_x, teste_y)],
                verbose=100)
ypred_xgb = classif_xgb.predict(teste_x)
score_xgb = matthews_corrcoef(teste_y.convert_dtypes('numeric'), ypred_xgb)
print(f'Score Sklearn MCC local: {score_xgb:.5f}')

# Record the informations regarding the model and score in local files
registro_geral = pd.read_csv('registros/registros_resultados.csv')
registro_atual = pd.DataFrame([[pd.to_datetime(nome_modelo), modelo_xgb, round(score_xgb, 5)]])
registro_atual.columns = ('Dia e Hora', 'Modelo', 'Score (MCC)')
if registro_geral.iloc[registro_geral.shape[0]-1]['Dia e Hora'] == str(pd.to_datetime(nome_modelo)):
    print('Trabalhando com mesmo modelo')
else:
    realiza_registro_xgb()

# Estimates the challenge values, creating the file for submission on Kaggle
y_xgb_final = classif_xgb.predict(teste_na)
submissao = pd.read_parquet('datasets/submission.parquet')
submissao['class'] = y_xgb_final
submissao.to_csv('submissoes/submissao_xgbclassifier_'+nome_modelo+'.csv', index=False)


#########################################################################
#                                                                       #
#                        BEGINNING OF TEST AREA                         #
#                                                                       #
#########################################################################

# A imputação também pode ser realizada através de um modelo de classificação multiclasse,
# faz um split de cada dataset, treina individualmente e compara. Se for bom, procede com
# a imputação através do modelo.

# Testando imputação dos valores NaN através de KNN para um melhor score
(teste.isnull().sum()/teste.shape[0])*100
(treino.isnull().sum()/treino.shape[0])*100

imputer = KNNImputer(weights='distance',
                     copy=False)
imputer.fit_transform(X)

#########################################################################
#                                                                       #
#                           END OF TEST AREA                            #
#                                                                       #
#########################################################################


###
# AREA BELOW HERE JUST TO HELP MY WORKFLOW
###

treino.to_parquet('datasets/treino_polido.parquet')
teste.to_parquet('datsets/teste_polido.parquet')

treino_y = treino_yback.copy()
teste_y = teste_yback.copy()

registro_geral = pd.read_csv('registros/registros_resultados.csv')
registro_geral
