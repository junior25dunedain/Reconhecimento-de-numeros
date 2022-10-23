import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

# carregando os dados
digitos = datasets.load_digits()
print(digitos)

# olhando algumas imagens
images_labels = list(zip(digitos.images,digitos.target))
for index, (ima,label) in enumerate(images_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(ima,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title(f'Label: {label}')
plt.show()

# separando dados de entrada e saida
x = digitos.data
y = digitos.target

print(x.shape,y.shape,'\n')

# dividindo dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.3,random_state=101)

treinodata, validata, treinolabel, validlabel = train_test_split(x_treino,y_treino,test_size=0.1,random_state=84)

print(f'Exemplos de treino: {len(treinolabel)}')
print(f'Exemplos de Validação: {len(validlabel)}')
print(f'Exemplos de teste: {len(y_teste)}')

# normalizando os dados pela média
X_norm = np.mean(x,axis=0)
x_treino_norm = treinodata - X_norm
x_valid_norm = validata - X_norm
x_teste_norm = x_teste - X_norm

print(x_treino_norm.shape,x_valid_norm.shape,x_teste_norm.shape)

# Determinando o melhor modelo classificador KNeighbors
kval = range(1,18,2)

acuracia = []

for k in kval:
    modeloKNN = KNeighborsClassifier(n_neighbors=k)
    modeloKNN.fit(treinodata,treinolabel)

    score = modeloKNN.score(validata,validlabel)
    print(f'Com o valor de k={k}, a acurácia é = {score*100:.2f}')
    acuracia.append(score)

# obtendo o valor de vizinhanças que gerou o modelo mais eficiente
f = np.argmax(acuracia)

# criando o modelo final KNeighborsClassifier

modeloKNN_final = KNeighborsClassifier(n_neighbors=kval[f])
modeloKNN_final.fit(treinodata,treinolabel)

previsao = modeloKNN_final.predict(x_teste)
print('Avaliação do modelo nos dados de teste')
print(classification_report(y_teste,previsao))

print('matrix da confusão')
print(confusion_matrix(y_teste,previsao))

# fazendo previsoes
for i in np.random.randint(0,high=len(y_teste),size=(5,)):
    image = x_teste[i]
    preditos = modeloKNN_final.predict([image])[0]

    imgdata = np.array(image,dtype='float')
    pixels = imgdata.reshape((8,8))
    plt.imshow(pixels,cmap='gray')
    plt.annotate(preditos,(3,3),bbox={'facecolor':'white'},fontsize=16)
    print(f'Eu acredito que esse digito seja: {preditos}')
    plt.show()







