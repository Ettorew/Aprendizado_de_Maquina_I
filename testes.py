from ID3 import *
from C45 import *
from utils_classificacao import *
import pandas as pd

df = pd.read_csv("weather.nominal.csv")
coluna_alvo = 'play'
atributos = [c for c in df.columns if c != coluna_alvo]

treino, teste = treino_teste(df, 0.7)

ID3 = NoArvoreID3.construir_arvore(treino, atributos, coluna_alvo)

C45 = NoArvoreC45.construir_arvore(treino, atributos, coluna_alvo)

previsoes_ID3 = prever(teste, ID3)

previsoes_C45 = prever(teste, C45)

y = teste['play'].tolist()

resultado_ID3 = metricas(y, previsoes_ID3)

resultado_C45 = metricas(y, previsoes_C45)

print("ID3: " + resultado_ID3)

print("C45: " + resultado_C45)