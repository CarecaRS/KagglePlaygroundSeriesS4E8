###
# WORK NOTES (a.k.a. 'to-do list')
###
# - The parquet files (treino/teste_polido) have only categorical NaN values, all other information is already cleaned/set up.
# - The categorical NaN values must have a proper imputation

# Importing necessary packages
import pandas as pd
import numpy as np
import os
#import tensorflow as tf
import io
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import normalize
pd.options.display.float_format = '{:.4f}'.format
%autoindent

# Some functions for recording the models development
def registro_modelo(model):
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
        registros.write(registro_modelo(modelo))
        registros.write("   --> Score local do modelo (accuracy): " + str(score))
        registros.write("\n\n########## FINAL DE REGISTRO - MODELO TensorFlow " + nome_modelo + " ##########\n\n\n\n")
    print('Novo registro realizado com sucesso!')


# Defines my workspace
os.chdir('/home/thiago/Documentos/MBA USP/Kaggle/Mushrooms/')

# Database import
#teste_completo = pd.read_parquet('test.parquet')
#treino_completo = pd.read_parquet('train.parquet')
#teste = teste_completo.copy()
#treino = treino_completo.copy()

teste = pd.read_parquet('teste_polido.parquet')
treino = pd.read_parquet('treino_polido.parquet')

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


###
# ÁREA DE TESTES
###

# Normalizing the numeric values just before modeling, so I'm able to test
# the results with different normalization hyperparameters
numerics = treino.columns[treino.dtypes == 'float64']
norm = 'l1'  # 'l1' or 'l2'
treino_na[numerics] = normalize(treino_na[numerics], norm=norm)
teste_na[numerics] = normalize(teste_na[numerics], norm=norm)

# Train/test split
tamanho_treino = 0.80
target = 'class'
treino_x, teste_x, treino_y, teste_y = train_test_split(treino_na, teste_na,
                                                        train_size=tamanho_treino,
                                                        random_state=1)





















treino.to_parquet('treino_polido.parquet')
teste.to_parquet('teste_polido.parquet')
