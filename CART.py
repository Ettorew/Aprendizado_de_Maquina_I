import math
import pandas as pd

def indice_gini(coluna_alvo):
    """
    Calcula o índice Gini de uma coluna de classes.
    Gini = 1 - soma(p_i²)
    """
    proporcoes = coluna_alvo.value_counts(normalize=True)
    return 1 - sum(proporcoes ** 2)

def gini_condicional(dataset, atributo, coluna_alvo):
    gini_total = 0
    total = len(dataset)
    for valor in dataset[atributo].unique():
        subset = dataset[dataset[atributo] == valor]
        proporcao = len(subset) / total
        gini_total += proporcao * indice_gini(subset[coluna_alvo])
    return gini_total

def calcular_gini(dataset, atributo, coluna_alvo):
    gini_inicial = indice_gini(dataset[coluna_alvo])
    gini_após_divisao = gini_condicional(dataset, atributo, coluna_alvo)
    return gini_inicial - gini_após_divisao


def melhor_limite_numerico(dataset, atributo, coluna_alvo):
    """
    Retorna o melhor ponto de corte para um atributo numérico baseado no índice Gini.
    Trabalha diretamente com splits binários (≤ limite e > limite) sem criar strings.
    """
    # Ordena os valores
    dataset_ordenado = dataset.sort_values(by=atributo)
    valores = dataset_ordenado[atributo].values
    classes = dataset_ordenado[coluna_alvo].values

    # Inicializa
    melhor_limite = None
    melhor_score = -float("inf")
    gini_inicial = indice_gini(dataset[coluna_alvo])

    # Possíveis limiares entre valores consecutivos com classes diferentes
    candidatos = [(valores[i] + valores[i-1]) / 2
                  for i in range(1, len(valores))
                  if classes[i] != classes[i-1]]

    for limite in candidatos:
        # Subsets do split binário
        subset_le = dataset[dataset[atributo] <= limite]
        subset_gt = dataset[dataset[atributo] > limite]

        # Gini ponderado
        gini_split = (len(subset_le) / len(dataset)) * indice_gini(subset_le[coluna_alvo]) + \
                     (len(subset_gt) / len(dataset)) * indice_gini(subset_gt[coluna_alvo])

        ganho_gini = gini_inicial - gini_split

        if ganho_gini > melhor_score:
            melhor_score = ganho_gini
            melhor_limite = limite

    return melhor_limite, melhor_score

class NoArvoreCART:
    def __init__(self, atributo=None, limite=None, filhos=None, classe=None):
        self.atributo = atributo
        self.limite = limite
        self.filhos = filhos if filhos is not None else {}
        self.classe = classe

class ArvoreCART:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.raiz = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def treinar(self, dataset, atributos, coluna_alvo):
        self.raiz = self._construir_arvore(dataset, atributos, coluna_alvo, 0)

    def _construir_arvore(self, dataset, atributos, coluna_alvo, depth):
        # Casos base
        if dataset.empty:
            raise ValueError("Dataset vazio na construção da árvore")
        if len(dataset[coluna_alvo].unique()) == 1:
            return NoArvoreCART(classe=dataset[coluna_alvo].iloc[0])
        if not atributos or (self.max_depth is not None and depth >= self.max_depth) \
            or len(dataset) <= self.min_samples_leaf:
            return NoArvoreCART(classe=dataset[coluna_alvo].mode()[0])

        # Escolher melhor atributo
        melhor_score = -float("inf")
        melhor_atributo = None
        melhor_limite = None

        for atributo in atributos:
            if dataset[atributo].dtype.kind in 'bifc':
                limite, score = self._melhor_limite_numerico(dataset, atributo, coluna_alvo)
            else:
                score = self._calcular_gini(dataset, atributo, coluna_alvo)
                limite = None
            if score > melhor_score:
                melhor_score = score
                melhor_atributo = atributo
                melhor_limite = limite

        if melhor_atributo is None:
            return NoArvoreCART(classe=dataset[coluna_alvo].mode()[0])

        no = NoArvoreCART(atributo=melhor_atributo, limite=melhor_limite)
        atributos_restantes = [a for a in atributos if a != melhor_atributo]

        # Subdivisão
        if melhor_limite is not None:
            subset_le = dataset[dataset[melhor_atributo] <= melhor_limite]
            subset_gt = dataset[dataset[melhor_atributo] > melhor_limite]

            no.filhos["<="] = self._construir_arvore(subset_le, atributos_restantes, coluna_alvo, depth + 1) \
                if not subset_le.empty else NoArvoreCART(classe=dataset[coluna_alvo].mode()[0])
            no.filhos[">"] = self._construir_arvore(subset_gt, atributos_restantes, coluna_alvo, depth + 1) \
                if not subset_gt.empty else NoArvoreCART(classe=dataset[coluna_alvo].mode()[0])
        else:
            for valor in dataset[melhor_atributo].unique():
                subset = dataset[dataset[melhor_atributo] == valor]
                no.filhos[valor] = self._construir_arvore(subset, atributos_restantes, coluna_alvo, depth + 1)

        return no

    # ----------------------------
    # Cálculo do Gini
    # ----------------------------
    @staticmethod
    def _indice_gini(coluna):
        proporcoes = coluna.value_counts(normalize=True)
        return 1 - sum(proporcoes ** 2)

    @staticmethod
    def _calcular_gini(dataset, atributo, coluna_alvo):
        gini_total = 0
        total = len(dataset)
        for valor in dataset[atributo].unique():
            subset = dataset[dataset[atributo] == valor]
            gini_total += len(subset) / total * ArvoreCART._indice_gini(subset[coluna_alvo])
        return ArvoreCART._indice_gini(dataset[coluna_alvo]) - gini_total

    @staticmethod
    def _melhor_limite_numerico(dataset, atributo, coluna_alvo):
        ordenado = dataset.sort_values(by=atributo)
        valores = ordenado[atributo].values
        classes = ordenado[coluna_alvo].values
        candidatos = [(valores[i] + valores[i-1])/2 for i in range(1, len(valores)) if classes[i] != classes[i-1]]

        melhor_score = -float("inf")
        melhor_limite = None
        for lim in candidatos:
            subset_le = dataset[dataset[atributo] <= lim]
            subset_gt = dataset[dataset[atributo] > lim]
            score = ArvoreCART._indice_gini(dataset[coluna_alvo]) - (
                len(subset_le)/len(dataset) * ArvoreCART._indice_gini(subset_le[coluna_alvo]) +
                len(subset_gt)/len(dataset) * ArvoreCART._indice_gini(subset_gt[coluna_alvo])
            )
            if score > melhor_score:
                melhor_score = score
                melhor_limite = lim
        return melhor_limite, melhor_score

    # ----------------------------
    # Poda custo-complexidade
    # ----------------------------
    def poda(self, dataset, coluna_alvo, alpha=0.0):
        self.raiz = self._poda_subarvore(self.raiz, dataset, coluna_alvo, alpha)

    def _poda_subarvore(self, no, dataset, coluna_alvo, alpha):
        if no.classe is not None:
            return no

        # Poda recursiva nos filhos
        for valor, filho in list(no.filhos.items()):
            if no.limite is not None:
                subset = dataset[dataset[no.atributo] <= no.limite] if valor == "<=" else dataset[dataset[no.atributo] > no.limite]
            else:
                subset = dataset[dataset[no.atributo] == valor]
            no.filhos[valor] = self._poda_subarvore(filho, subset, coluna_alvo, alpha)

        # Erro da subárvore
        erro_total = self._erro_subarvore(no, dataset, coluna_alvo)
        classe_mais_comum = dataset[coluna_alvo].mode()[0]
        erro_no = sum(dataset[coluna_alvo] != classe_mais_comum)

        n_folhas = self._conta_folhas(no)
        if n_folhas > 1 and (erro_no + alpha) <= (erro_total + alpha * n_folhas):
            return NoArvoreCART(classe=classe_mais_comum)
        return no

    def _erro_subarvore(self, no, dataset, coluna_alvo):
        if no.classe is not None:
            return sum(dataset[coluna_alvo] != no.classe)
        erro = 0
        for valor, filho in no.filhos.items():
            if no.limite is not None:
                subset = dataset[dataset[no.atributo] <= no.limite] if valor == "<=" else dataset[dataset[no.atributo] > no.limite]
            else:
                subset = dataset[dataset[no.atributo] == valor]
            erro += self._erro_subarvore(filho, subset, coluna_alvo)
        return erro

    def _conta_folhas(self, no):
        if no.classe is not None:
            return 1
        return sum(self._conta_folhas(filho) for filho in no.filhos.values())

    # ----------------------------
    # Impressão
    # ----------------------------
    def imprimir_arvore(self, no=None, nivel=0):
        no = no or self.raiz
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
            self.imprimir_arvore(filho, nivel + 1)


