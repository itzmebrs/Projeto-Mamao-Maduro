# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:35:58 2020

@author: A552698
"""

import cv2
from PIL import Image
import numpy as np
from pylab import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import os #Permissão para acesso ao sistema operacional (pastas)
import imFerramentas as imF
import imutils
import csv
from extractor_gray import extract_gray_features
import glob
import sys
import seaborn as sns

input_folder_tratado = 'mamaocompleto/mamao-tratado' #import das imagens tratadas
output_file = 'dataset_mamao.csv' #Arquivo com o Dataset dos descritores de cor dos mamãos
files_tratado = [os.path.join(input_folder_tratado, f) for f in os.listdir(input_folder_tratado)]

#Trabalhando com os Datasets

csv_complete = [] #Declarando o Dataset que será criado
linha_csv = [] #variável de apoio para criar as linhas do dataset
#lendo os datasets

#Criando a estrutura do Dataset
#nome das features que serão extraídas das imagens tratadas
gray_features_names = ['mean_I','std_I','entropy_I','std_hist_I','kurt_hist_I','skew_hist_I',
                       'lbp_0','lbp_1','lbp_2','lbp_3','lbp_4','lbp_5','lbp_6','lbp_7',
                       'lbp_8','lbp_9','com_entropy','com_inertia','com_energy',
                       'com_correlation','com_homogeniety','FFT_energy','FFT_entropy',
                       'FFT_intertia','FFT_homogeneity','SNM','EME']
#concatenando o nome do arquivo e a classe pertencente
features_names = ['Name_file'] + gray_features_names + ['Class']

csv_complete.append(features_names)

#Lendo as imagens, extaindo os descritores de cor e salvando os valores no Dataset
for index,arquivo in enumerate(files_tratado):
    nome_arquivo_completo = os.path.basename(arquivo[28:]) #Pegando o nome do arquivo
    if(cv2.imread(arquivo) is not None):
        img = cv2.imread(arquivo,0)
        linha_csv = [nome_arquivo_completo]
        linha_csv = linha_csv + extract_gray_features(img) #Concatenando os descritores de cor
        linha_csv = linha_csv + [os.path.basename(arquivo[39:42])] #Concatenando a classe (grau de maturação)
        csv_complete.append(linha_csv)

with open(output_file, 'w') as myfile: #Salvando o dataset em um arquivo CSV
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(csv_complete)

#Avaliando as Features
arquivo = 'dataset_mamao.csv'
df = []
#lendo o arquivo na variável df (dataFrame) e setando os nomes das colunas
df_original = pd.read_csv(arquivo, names = ['Name_file','mean_I','std_I','entropy_I','std_hist_I',
                                   'kurt_hist_I','skew_hist_I','lbp_0','lbp_1','lbp_2',
                                   'lbp_3','lbp_4','lbp_5','lbp_6','lbp_7','lbp_8','lbp_9',
                                   'com_entropy','com_inertia','com_energy','com_correlation',
                                   'com_homogeniety','FFT_energy','FFT_entropy',
                                   'FFT_intertia','FFT_homogeneity','SNM','EME','Class'])
df = df_original[1:]
cols = ['mean_I','std_I','entropy_I','std_hist_I',
        'kurt_hist_I','skew_hist_I','lbp_0','lbp_1','lbp_2',
        'lbp_3','lbp_4','lbp_5','lbp_6','lbp_7','lbp_8','lbp_9',
        'com_entropy','com_inertia','com_energy','com_correlation',
        'com_homogeniety','FFT_energy','FFT_entropy',
        'FFT_intertia','FFT_homogeneity','SNM','EME']

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

#Trabalhando com o Dataset Gerado

#Plot do DistPlot
fig = plt.figure(figsize=(24, 20))
aux = 1
for atrib in cols:
    a = fig.add_subplot(9, 3, aux)
    sns.distplot(df[df.Class=="EM1"][atrib], label='EM1', color = 'g')
    sns.distplot(df[df.Class=="EM2"][atrib], label='EM2', color = 'y')
    sns.distplot(df[df.Class=="EM3"][atrib], label='EM3', color = 'r')
    tight_layout()
    aux = aux+1

fig = plt.figure(figsize=(24, 20))
aux = 1
for atrib in cols:
    atrib ="com_correlation"
    a = fig.add_subplot(5, 6, aux)
    sns.boxplot(data=df, x="Class", y=atrib)
    a.set_title(atrib)
    a.set_xlabel('')
    a.set_ylabel('')
    aux = aux + 1

resultado = []#array para guardar os resultados após aplicação da lógica
resultado_heads = ['File']+['Class']+['Result'] 
resultado.append(resultado_heads)
i = 0
for i in range(0,len(df)):
    if df.iloc[i][20]>0.9994:
        new_line = [df.iloc[i][0]]
        new_line = new_line + [df.iloc[i][28]]
        new_line = new_line +["EM3"]
        resultado.append(new_line)
    elif df.iloc[i][26] >0.000002 and df.iloc[i][26] < 0.000008:
        new_line = [df.iloc[i][0]]
        new_line = new_line + [df.iloc[i][28]]
        new_line = new_line +["EM2"]
        resultado.append(new_line)
    else:
        new_line = [df.iloc[i][0]]
        new_line = new_line + [df.iloc[i][28]]
        new_line = new_line +["EM1"]
        resultado.append(new_line)

with open('mamao-resultado.csv', 'w') as myfile: #Salvando o resultado em um arquivo CSV
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(resultado)