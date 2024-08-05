# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import os
#import tensorflow as tf
import io
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
%autoindent

# Definição das funções necessárias para a modelagem
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


# Definição do diretório de trabalho
os.chdir('/home/thiago/Documentos/MBA USP/Kaggle/Mushrooms/')

# Importação dos datasets
teste = pd.read_parquet('test.parquet')
treino = pd.read_parquet('treino.parquet')

target = 'label'

# Separação x e y com normalização do x
dados_treino = treino.drop(target, axis=1)
dados_teste = treino[target]
dados_treino = dados_treino/255

# Separação dos dados de treino entre treino e validação
tamanho_treino = 0.80
treino_x, teste_x, treino_y, teste_y = train_test_split(dados_treino, dados_teste,
                                                        train_size=tamanho_treino,
                                                        random_state=1)
