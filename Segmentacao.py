# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:31:16 2020

@author: A552698
"""

import cv2
from PIL import Image
from numpy import *
from pylab import *
import pandas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import os #Permissão para acesso ao sistema operacional (pastas)
import imFerramentas as imF
import imutils
import csv
from extractor_gray import extract_gray_features
import glob
import sys


input_folder_original = 'mamaocompleto' #Import das imagens originais
files_original = [os.path.join(input_folder_original, f) for f in os.listdir(input_folder_original)]

input_folder_menor = 'mamaocompleto/mamao-menor' #import das imagens menores
files_menor = [os.path.join(input_folder_menor, f) for f in os.listdir(input_folder_menor)]

input_folder_tratado = 'mamaocompleto/mamao-tratado' #import das imagens tratadas
files_tratado = [os.path.join(input_folder_tratado, f) for f in os.listdir(input_folder_tratado)]



def reduzir_arquivos (arquivos,pct_red): #Função que reduz os arquivos em x% (informado na chamada)
    for index,arquivo in enumerate(arquivos):
        nome_arquivo_completo = os.path.basename(arquivo)
        if(cv2.imread(arquivo) is not None):
            image =  cv2.imread(arquivo)
            res = (int(image.shape[1]*pct_red/100),int(image.shape[0]*pct_red/100))
            image = cv2.resize(image,res,interpolation = cv2.INTER_AREA)
            cv2.imwrite('mamaocompleto/mamao-menor/60_'+arquivo[14:],image)

def seg_passo1(imagem_cv2): #Função para plotar uma imagem em RGB e HSV
    img1 = cv2.cvtColor(imagem_cv2,cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 5))
    a = fig.add_subplot(2, 3,1)
    plt.imshow(img1[:,:,0],cmap='Reds')
    a.axis('Off')
    a.set_title("Canal R")
    b = fig.add_subplot(2,3,2)
    plt.imshow(img1[:,:,1],cmap='Greens')
    b.axis('Off')
    b.set_title("Canal G")
    c = fig.add_subplot(2,3,3)
    plt.imshow(img1[:,:,2],cmap='Blues')
    c.axis('Off')
    c.set_title("Canal B")
    
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    fig = plt.figure(figsize=(10, 5))
    a = fig.add_subplot(2, 3,1)
    plt.imshow(img1[:,:,0],cmap='gray')
    a.axis('Off')
    a.set_title("Canal H")
    b = fig.add_subplot(2,3,2)
    plt.imshow(img1[:,:,1],cmap='gray')
    b.axis('Off')
    b.set_title("Canal S")
    c = fig.add_subplot(2,3,3)
    plt.imshow(img1[:,:,2],cmap='gray')
    c.axis('Off')
    c.set_title("Canal V")

def segmentacao(img): #Função para aplicar o algoritmo de segmentação Espera-se receber a imagem em RGB
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17)) #Serve para operações Morfológicas
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #Convertendo a imagem para HSV
    img_threshold = imF.limiarizacao_janela(img_hsv[:,:,1], 100,260)
    img_blur = cv2.medianBlur(img_threshold, 43)
    img_erode = cv2.erode(img_blur, element, iterations = 3)
    img_dilate = cv2.dilate(img_erode, element, iterations = 3)
    #Tratamento para o Flood Fill
    h, w = img_dilate.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill = img_dilate.copy()
    #Aplicando o flood fill
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill) #bitwise
    img_floodfill = img_dilate | im_floodfill_inv
    img_final = cv2.bitwise_and(img,img,mask=img_floodfill)
    return img_final

#Desenvolvimento do algoritmo de segmentação para 1 amostra
'''
input_folder = 'mamaocompleto/Mamaotratado'
files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
dataset = pandas.DataFrame()
element = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17)) #Serve para operações Morfológicas

image = cv2.imread(files[5]) #Abrindo a imagem em CV2 (BGR)
image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convertendo a imagem original pra RGB
img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #Convertendo a imagem para HSV


#1º Tratamento -> Isolar o canal S
img_s = img_hsv[:,:,1]

#2º Tratamento -> Analisar o Histograma para definir o Limiar
histograma= plt.hist(img_s.flatten(),128)
plt.clf() #apenas para não mostrar o histograma e bagunçar o output

#3º Tratamento -> Aplicar o limiar na imagem utilizando o iMFerramentas
img_threshold = imF.limiarizacao_janela(img_s, 100,260)

#4º Tratamento -> Aplicar filtro Passa-Baixa para suavizar as imperfeições
img_blur = cv2.medianBlur(img_threshold, 43)

#5º Tratamento -> Aplicação de erosão para remover as informações do marcador
img_erode = cv2.erode(img_blur, element, iterations = 3)

#6º Tratamento -> Aplicação de dilatação para recuperar as informações perdidas na Erosão
img_dilate = cv2.dilate(img_erode, element, iterations = 3)

#7º Tratamento -> Aplicação de Flood Fill para preencher buracos em algumas amostras
#Tratamento para o Flood Fill
h, w = img_dilate.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
im_floodfill = img_dilate.copy()
#Aplicando o flood fill
cv2.floodFill(im_floodfill, mask, (0,0), 255);
im_floodfill_inv = cv2.bitwise_not(im_floodfill) #bitwise
img_floodfill = img_dilate | im_floodfill_inv

#8º Tratamento -> Aplicando a máscara final na imagem original
img_final = cv2.bitwise_and(image_original,image_original,mask=img_floodfill)
'''
#FIM

#Plotando os resultados
'''
fig = plt.figure(figsize=(16, 10))

plot_img_s = fig.add_subplot(2,4,1)
plot_img_s.set_title("Imagem no canal S")
plt.imshow(img_s, cmap='gray')
plt.axis('Off')

plot_img_s_hist = fig.add_subplot(2,4,2)
plot_img_s_hist.set_title("Histograma o canal S")
plt.hist(img_s.flatten(),128)

plot_img_threshold = fig.add_subplot(2,4,3)
plot_img_threshold.set_title("Limiarização - 100~260")
plt.imshow(img_threshold, cmap='gray')
plt.axis('Off')

plot_img_blur = fig.add_subplot(2,4,4)
plot_img_blur.set_title("Filtro de Suavização (Passa-baixa)")
plt.imshow(img_blur, cmap='gray')
plt.axis('Off')

plot_img_erode = fig.add_subplot(2,4,5)
plot_img_erode.set_title("Erosão - 3 Iterações")
plt.imshow(img_erode, cmap='gray')
plt.axis('Off')

plot_img_dilate = fig.add_subplot(2,4,6)
plot_img_dilate.set_title("Dilatação - 3 Iterações")
plt.imshow(img_dilate, cmap='gray')
plt.axis('Off')

plot_img_floodfill = fig.add_subplot(2,4,7)
plot_img_floodfill.set_title("Flood Fill")
plt.imshow(img_floodfill, cmap='gray')
plt.axis('Off')

plot_img_final = fig.add_subplot(2,4,8)
plot_img_final.set_title("Imagem original com Tratamento")
plt.imshow(img_final, cmap='gray')
plt.axis('Off')
'''
#FIM

#Aplicação do algoritmo de Segmentação em todas as amostras e salvando-as em uma pasta
for index,arquivo in enumerate(files_menor):
    nome_arquivo_completo = os.path.basename(arquivo)
    if(cv2.imread(arquivo) is not None):
        image =  cv2.imread(arquivo)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = segmentacao(image)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('mamaocompleto/mamao-tratado/isolado_'+arquivo[26:],image)

