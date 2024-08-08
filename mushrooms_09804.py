###
# NOTES
###
# This piece of code was able to score 0.9804 in the leaderbord, through CatBoost algorithm.
# There's a little work to do still, like performing a proper imputation for the NaN observations.
# I'm just making this code available as a final version so I can work on the other approaches. 
# - On CatBoost model I've already tested grow_policy (best Depth), bootstrap_type (best MVS) and I was testing learning_rate (best 0.081 so far)

# Importing necessary packages
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import normalize
from catboost import CatBoostClassifier, Pool
pd.options.display.float_format = '{:.4f}'.format

# I work with Neovim using Ipython, so the autoindent needs to be set OFF
%autoindent

# Create a function to record the model parameters and score in separated files
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


# Defining my workspace
os.chdir('/home/thiago/Documentos/MBA USP/Kaggle/Mushrooms/')

# Database import (both datasets were already cleaned up and are ready to use)
teste = pd.read_parquet('datasets/teste_cat.parquet')
treino = pd.read_parquet('datasets/treino_cat.parquet')

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
treino_na = treino_na.drop('id', axis=1)
teste_na = teste_na.drop('id', axis=1)
tamanho_treino = 0.80
target = 'class'

treino_x, teste_x, treino_y, teste_y = train_test_split(treino_na.drop(target, axis=1), treino_na[target],
                                                        train_size=tamanho_treino,
                                                        random_state=1)

###
# MODELLING WITH CATBOOST CLASSIFIER
###

# CatBoost needs the target variable to be binary (1 or 0), so the backup and transformations below
treino_yback = treino_y.copy()
teste_yback = teste_y.copy()
treino_y[treino_y == 'e'] = 1
treino_y[treino_y == 'p'] = 0
teste_y[teste_y == 'e'] = 1
teste_y[teste_y == 'p'] = 0

# The model itself
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")
modelo_cat = 'CatBoostClassifier'
catboost = CatBoostClassifier(cat_features=categories,
                              loss_function='CrossEntropy',
                              eval_metric='MCC',
                              iterations=1000,
                              learning_rate=0.081,
                              random_seed=1,
                              bootstrap_type='MVS',
                              bagging_temperature=7,
                              depth=16,
                              early_stopping_rounds=200,
                              thread_count=12,
                              task_type='CPU',
                              target_border=0.51,
                              grow_policy='Depthwise',
                              min_child_samples=39,
                              boosting_type='Plain'
                              )

# Makes a pool for both train and test objects, using the raw categorical features
pool_treino = Pool(treino_x,
                   label=treino_y,
                   cat_features=categories)
pool_teste = Pool(teste_x,
                  label=teste_y,
                  cat_features=categories)

# Let's train the model
catboost.fit(pool_treino,
             eval_set=(pool_teste),
             verbose=100)

# And then predict our values and check our score
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
y_cat_final = catboost.predict(teste_na)
submissao = pd.read_parquet('datasets/submission.parquet')
submissao['class'] = y_cat_final
mask1 = submissao['class'] == 1
mask0 = submissao['class'] == 0
submissao.loc[mask1, 'class'] = 'e'
submissao.loc[mask0, 'class'] = 'p'
submissao.to_csv('submissoes/submissao_catboost_'+nome_modelo+'.csv', index=False)
