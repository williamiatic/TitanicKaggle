import pandas as pd
import numpy as np
base = pd.read_csv('train.csv') # Importando base de Treinamento
base = base.drop('Cabin',axis=1) # Tirando a Coluna Cabine
#base = base.drop('Ticket',axis=1)
base = base.drop('PassengerId',axis=1) # Tirando a Coluna ID Do Passageiro
#base = base.drop('Name',axis=1)
base.isnull().sum()

base['Age'].fillna(base['Age'].mean(), inplace=True) # Remove NaN da Coluna Age
base['Embarked'].fillna(base['Embarked'].mode()[0], inplace=True) # Remove NaN da Coluna Embarked


#Separando a tabela entre Previsores e Classe
previsores = base.iloc[:, 1:11].values
classe = base.iloc[:, 0].values


#Transformando Colunas de Texto para Numeros
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:,2])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:,8])


#Fazendo Scalonamento da base de dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.10, random_state=0)


## Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
# Naive Bayes

#Arvore de Decisão
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
# Arvore de Decisão

#Florestas Randomicas #Random Forest

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Florestas Randomicas #Random Forest


# Regras

#KNN

from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#KNN

#Regressão Logistica

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(random_state = 0)
classificador.fit(previsores, classe)

#Regressão Logistica

#Redes Neurais

from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True, max_iter = 1000, tol = 0.0000010)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#Redes Neurais
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)












from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

resultados30 = []
for i in range(30):
    
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros(shape=(previsores.shape[0], 1))):
        classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento]) 
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
    
resultados30 = np.asarray(resultados30)
for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))
