import math
import pandas as pd

def calcular_entropia(coluna_alvo):
    valores = coluna_alvo.value_counts(normalize=True)
    entropia = -sum(p * math.log2(p) for p in valores)
    return entropia

def entropia_condicional(dataset, atributo, coluna_alvo):
    entropia_total = 0
    total = len(dataset)
    
    # para cada valor único do atributo
    for valor in dataset[atributo].unique():
        subset = dataset[dataset[atributo] == valor]
        proporcao = len(subset) / total
        entropia_subset = calcular_entropia(subset[coluna_alvo])
        entropia_total += proporcao * entropia_subset
        
    return entropia_total

def calcular_ganho(dataset, atributo, coluna_alvo):
    return calcular_entropia(dataset[coluna_alvo]) - entropia_condicional(dataset, atributo, coluna_alvo)

class ArvoreID3:
    def __init__(self):
        self.raiz = None  # raiz da árvore

    def treinar(self, dataset, atributos, coluna_alvo):
        """Constrói a árvore e armazena a raiz"""
        self.raiz = self.construir_arvore(dataset, atributos, coluna_alvo)

    @staticmethod
    def construir_arvore(dataset, atributos, coluna_alvo):
        # Caso base 1: todas as instâncias têm a mesma classe
        if len(dataset[coluna_alvo].unique()) == 1:
            return NoArvoreID3(classe=dataset[coluna_alvo].iloc[0])
        
        # Caso base 2: não há mais atributos
        if len(atributos) == 0:
            classe_mais_comum = dataset[coluna_alvo].mode()[0]
            return NoArvoreID3(classe=classe_mais_comum)
        
        # Melhor atributo pelo ganho
        ganhos = {atributo: calcular_ganho(dataset, atributo, coluna_alvo) for atributo in atributos}
        melhor_atributo = max(ganhos, key=ganhos.get)
        
        no = NoArvoreID3(atributo=melhor_atributo)
        for valor in dataset[melhor_atributo].unique():
            subset = dataset[dataset[melhor_atributo] == valor]
            atributos_restantes = [a for a in atributos if a != melhor_atributo]
            no.filhos[valor] = ArvoreID3.construir_arvore(subset, atributos_restantes, coluna_alvo)
        return no

    @staticmethod
    def imprimir_arvore_no(no, nivel=0):
        indent = "  " * nivel
        if no.classe is not None:
            print(f"{indent}Folha: classe = {no.classe}")
            return
        print(f"{indent}Atributo: {no.atributo}")
        for valor, filho in no.filhos.items():
            print(f"{indent}-> Valor: {valor}")
            ArvoreID3.imprimir_arvore_no(filho, nivel + 1)

    def imprimir_arvore(self):
        if self.raiz is None:
            print("Árvore não construída!")
        else:
            self.imprimir_arvore_no(self.raiz)

# Classe interna para os nós da árvore
class NoArvoreID3:
    def __init__(self, atributo=None, filhos=None, classe=None):
        self.atributo = atributo
        self.filhos = filhos or {}
        self.classe = classe

