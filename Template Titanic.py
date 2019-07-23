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
