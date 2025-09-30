from ID3 import *
import pandas as pd

df = pd.read_csv("weather.nominal.csv")
coluna_alvo = 'play'
atributos = [c for c in df.columns if c != coluna_alvo]

treino, teste = treino_teste(df, 0.7)

arvore = construir_arvore(treino, atributos, coluna_alvo)

previsoes = prever(teste, arvore)

y = teste['play'].tolist()

resultado = metricas(y, previsoes)

print(resultado)