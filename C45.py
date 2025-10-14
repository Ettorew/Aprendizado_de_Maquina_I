import math
import pandas as pd

def calcular_entropia(coluna_alvo):
    valores = coluna_alvo.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in valores if p > 0)

def entropia_condicional(dataset, atributo, coluna_alvo):
    entropia_total = 0
    total = len(dataset)
    
    for valor in dataset[atributo].unique():
        subset = dataset[dataset[atributo] == valor]
        proporcao = len(subset) / total
        entropia_total += proporcao * calcular_entropia(subset[coluna_alvo])
        
    return entropia_total

def calcular_ganho(dataset, atributo, coluna_alvo):
    entropia_inicial = calcular_entropia(dataset[coluna_alvo])
    entropia_attr = entropia_condicional(dataset, atributo, coluna_alvo)
    return entropia_inicial - entropia_attr

def calcular_razao_ganho(dataset, atributo, coluna_alvo):
    total = len(dataset)
    
    # 🔹 Calcula o ganho de informação padrão
    ganho = calcular_ganho(dataset, atributo, coluna_alvo)
    
    # 🔹 Calcula a informação intrínseca (split info)
    info_atributo = 0
    for valor in dataset[atributo].unique():
        proporcao = len(dataset[dataset[atributo] == valor]) / total
        if proporcao > 0:
            info_atributo -= proporcao * math.log2(proporcao)
    
    # 🔹 Evita divisão por zero
    if info_atributo == 0:
        return 0
    
    # 🔹 Razão de ganho
    return ganho / info_atributo

def melhor_limite_numerico(dataset, atributo, coluna_alvo, usar_razao=False):
    """
    Retorna o melhor ponto de corte para um atributo numérico
    com base no ganho de informação ou razão de ganho.
    """
    # Ordena os valores
    dataset_ordenado = dataset.sort_values(by=atributo)
    valores = dataset_ordenado[atributo].values
    classes = dataset_ordenado[coluna_alvo].values

    # Possíveis limiares entre valores consecutivos com classes diferentes
    candidatos = [(valores[i] + valores[i-1]) / 2 
                  for i in range(1, len(valores)) 
                  if classes[i] != classes[i-1]]

    melhor_limite = None
    melhor_score = -float("inf")

    for limite in candidatos:
        # Cria coluna temporária sem alterar a original
        dataset_temp = dataset.copy()
        dataset_temp['_tmp'] = dataset_temp[atributo].apply(
            lambda x: f"≤{limite}" if x <= limite else f">{limite}"
        )

        # Calcula score
        score = calcular_razao_ganho(dataset_temp, '_tmp', coluna_alvo) if usar_razao else calcular_ganho(dataset_temp, '_tmp', coluna_alvo)

        if score > melhor_score:
            melhor_score = score
            melhor_limite = limite

    return melhor_limite, melhor_score

class ArvoreC45:
    def __init__(self):
        self.raiz = None  # raiz da árvore

    def treinar(self, dataset, atributos, coluna_alvo):
        """Constrói a árvore e armazena a raiz"""
        self.raiz = self.construir_arvore(dataset, atributos, coluna_alvo)

    @staticmethod
    def construir_arvore(dataset, atributos, coluna_alvo):
        # Caso base 1: dataset vazio
        if dataset.empty:
            # Retorna nó folha com classe mais comum do dataset original
            raise ValueError("Dataset vazio na construção da árvore. Deve-se passar dataset não vazio.")

        # Caso base 2: todas as instâncias têm a mesma classe
        if len(dataset[coluna_alvo].unique()) == 1:
            return NoArvoreC45(classe=dataset[coluna_alvo].iloc[0])
        
        # Caso base 3: não há mais atributos
        if len(atributos) == 0:
            classe_mais_comum = dataset[coluna_alvo].mode()[0]
            return NoArvoreC45(classe=classe_mais_comum)
        
        # Escolher o melhor atributo pelo ganho ou razão de ganho
        melhor_score = -float("inf")
        melhor_atributo = None
        melhor_limite = None

        for atributo in atributos:
            if dataset[atributo].dtype.kind in 'bifc':  # numérico
                limite, score = melhor_limite_numerico(dataset, atributo, coluna_alvo, usar_razao=True)
            else:
                score = calcular_razao_ganho(dataset, atributo, coluna_alvo)
                limite = None

            if score > melhor_score:
                melhor_score = score
                melhor_atributo = atributo
                melhor_limite = limite

        no = NoArvoreC45(atributo=melhor_atributo, limite=melhor_limite)

        atributos_restantes = [a for a in atributos if a != melhor_atributo]

        # Subdivisão
        if melhor_limite is not None:
            # Numérico: cria dois subsets
            subset_esq = dataset[dataset[melhor_atributo] <= melhor_limite]
            subset_dir = dataset[dataset[melhor_atributo] > melhor_limite]

            # Nó folha se subset vazio
            if subset_esq.empty:
                classe_mais_comum = dataset[coluna_alvo].mode()[0]
                no.filhos[f"<= {melhor_limite:.3f}"] = NoArvoreC45(classe=classe_mais_comum)
            else:
                no.filhos[f"<= {melhor_limite:.3f}"] = ArvoreC45.construir_arvore(subset_esq, atributos_restantes, coluna_alvo)

            if subset_dir.empty:
                classe_mais_comum = dataset[coluna_alvo].mode()[0]
                no.filhos[f"> {melhor_limite:.3f}"] = NoArvoreC45(classe=classe_mais_comum)
            else:
                no.filhos[f"> {melhor_limite:.3f}"] = ArvoreC45.construir_arvore(subset_dir, atributos_restantes, coluna_alvo)

        else:
            # Categórico: cria subset para cada valor
            for valor in dataset[melhor_atributo].unique():
                subset = dataset[dataset[melhor_atributo] == valor]
                if subset.empty:
                    classe_mais_comum = dataset[coluna_alvo].mode()[0]
                    no.filhos[valor] = NoArvoreC45(classe=classe_mais_comum)
                else:
                    no.filhos[valor] = ArvoreC45.construir_arvore(subset, atributos_restantes, coluna_alvo)

        return no

    @staticmethod
    def imprimir_arvore_no(no, nivel=0):
        indent = "  " * nivel
        if no.classe is not None:
            print(f"{indent}Folha: classe = {no.classe}")
            return
        print(f"{indent}Atributo: {no.atributo}", end='')
        if no.limite is not None:
            print(f" (limite: {no.limite})")
        else:
            print()
        for valor, filho in no.filhos.items():
            print(f"{indent}-> Valor: {valor}")
            ArvoreC45.imprimir_arvore_no(filho, nivel + 1)

    def imprimir_arvore(self):
        if self.raiz is None:
            print("A árvore ainda não foi construída!")
        else:
            self.imprimir_arvore_no(self.raiz)

    @staticmethod
    def poda_c45(no, dataset, coluna_alvo):
        """
        Poda recursiva de uma árvore do tipo C4.5 (poda pessimista).
        """
        # Se for folha, nada a fazer
        if no.classe is not None:
            return no

        # Poda recursiva (bottom-up)
        for valor, filho in list(no.filhos.items()):
            if no.limite is not None:
                if valor.startswith("<="):
                    subset = dataset[dataset[no.atributo] <= no.limite]
                else:
                    subset = dataset[dataset[no.atributo] > no.limite]
            else:
                subset = dataset[dataset[no.atributo] == valor]

            no.filhos[valor] = ArvoreC45.poda_c45(filho, subset, coluna_alvo)

        # ---------------------------
        # Avaliar se deve podar o nó atual
        # ---------------------------
        classe_mais_comum = dataset[coluna_alvo].mode()[0]

        # Erro se o nó fosse folha
        erros_folha = sum(dataset[coluna_alvo] != classe_mais_comum)
        E_folha = (erros_folha + 0.5) / len(dataset)

        # Erro se mantivesse os filhos
        E_filhos = 0
        for valor, filho in no.filhos.items():
            if filho.classe is not None:
                if no.limite is not None:
                    if valor.startswith("<="):
                        subset = dataset[dataset[no.atributo] <= no.limite]
                    else:
                        subset = dataset[dataset[no.atributo] > no.limite]
                else:
                    subset = dataset[dataset[no.atributo] == valor]
                if len(subset) > 0:
                    erros = sum(subset[coluna_alvo] != filho.classe)
                    E_filhos += erros / len(dataset)

        # Comparar e decidir podar
        if E_folha <= E_filhos:
            return NoArvoreC45(classe=classe_mais_comum)
        else:
            return no


# Classe para os nós
class NoArvoreC45:
    def __init__(self, atributo=None, limite=None, filhos=None, classe=None):
        self.atributo = atributo
        self.limite = limite
        self.filhos = filhos or {}
        self.classe = classe
