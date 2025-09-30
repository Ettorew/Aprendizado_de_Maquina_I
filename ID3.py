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

class NoArvore:
    def __init__(self, atributo=None, filhos=None, classe=None):
        self.atributo = atributo   # atributo usado para divisão no nó
        self.filhos = filhos or {} # dicionário: valor do atributo -> subárvore
        self.classe = classe       # se for nó folha, armazena a classe

# Função recursiva para construir a árvore
def construir_arvore(dataset, atributos, coluna_alvo):
    # -----------------------
    # Caso base 1: todas as instâncias têm a mesma classe
    # -----------------------
    if len(dataset[coluna_alvo].unique()) == 1:
        return NoArvore(classe=dataset[coluna_alvo].iloc[0])
    
    # -----------------------
    # Caso base 2: não há mais atributos para dividir
    # -----------------------
    if len(atributos) == 0:
        # retorna a classe mais frequente
        classe_mais_comum = dataset[coluna_alvo].mode()[0]
        return NoArvore(classe=classe_mais_comum)
    
    # -----------------------
    # Caso recursivo: escolher o melhor atributo pelo ganho
    # -----------------------
    ganhos = {}
    for atributo in atributos:
        ganhos[atributo] = calcular_ganho(dataset, atributo, coluna_alvo)
    
    # melhor atributo = o que tem maior ganho
    melhor_atributo = max(ganhos, key=ganhos.get)
    
    # cria o nó com o melhor atributo
    no = NoArvore(atributo=melhor_atributo)
    
    # para cada valor do melhor atributo, cria um ramo recursivo
    for valor in dataset[melhor_atributo].unique():
        subset = dataset[dataset[melhor_atributo] == valor]
        # remove o atributo escolhido da lista para os nós filhos
        atributos_restantes = [a for a in atributos if a != melhor_atributo]
        # chama recursivamente
        no.filhos[valor] = construir_arvore(subset, atributos_restantes, coluna_alvo)
    
    return no

def prever_linha(exemplo, arvore):
    # Caso base: se for nó folha, retorna a classe
    if arvore.classe is not None:
        return arvore.classe
    
    # Caso recursivo: pega o valor do atributo no exemplo
    valor_atributo = exemplo[arvore.atributo]
    
    # Se existir um ramo para esse valor, segue recursivamente
    if valor_atributo in arvore.filhos:
        return prever_linha(exemplo, arvore.filhos[valor_atributo])
    else:
        # Se o valor não estiver nos ramos (ex: valor novo), podemos:
        # - retornar a classe mais comum dos filhos
        # - ou definir algum padrão (aqui, vamos pegar a primeira classe filha)
        # isso pode ser refinado conforme necessidade
        primeiro_filho = next(iter(arvore.filhos.values()))
        return prever_linha(exemplo, primeiro_filho)
    
def prever(df, arvore):
    # retorna uma lista com as classes previstas para cada linha
    previsoes = []
    for _, linha in df.iterrows():
        exemplo = linha.to_dict()
        previsoes.append(prever_linha(exemplo, arvore))
    return previsoes
    
def treino_teste(df, tam, frac=1, random_state=42):
    # Embaralhar o dataset
    df = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    # Determinar tamanho do treino
    tamanho_treino = int(tam * len(df))

    # Dividir
    train_df = df[:tamanho_treino]
    test_df = df[tamanho_treino:]
    return train_df, test_df

def metricas(y_true, y_pred):
    """
    y_true : lista ou série com valores reais
    y_pred : lista ou série com previsões
    Retorna: dicionário com acurácia, recall e F1-score
    """
    assert len(y_true) == len(y_pred), "Listas de tamanhos diferentes"

    # contadores
    tp = fp = tn = fn = 0

    for real, pred in zip(y_true, y_pred):
        if real == 'yes':
            if pred == 'yes':
                tp += 1
            else:
                fn += 1
        else:  # real == 'no'
            if pred == 'yes':
                fp += 1
            else:
                tn += 1

    # acurácia
    acuracia = (tp + tn) / (tp + tn + fp + fn)

    # recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # precisão = TP / (TP + FP)
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0

    # F1 = 2 * (precisão * recall) / (precisão + recall)
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

    return {"acuracia": acuracia, "recall": recall, "f1": f1}

