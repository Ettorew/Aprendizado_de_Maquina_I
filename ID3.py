import math
import pandas as pd

# Funções auxiliares (sem alteração)
def calcular_entropia(coluna_alvo):
    if coluna_alvo.empty:
        return 0
    valores = coluna_alvo.value_counts(normalize=True)
    entropia = -sum(p * math.log2(p) for p in valores if p > 0)
    return entropia

def entropia_condicional(dataset, atributo, coluna_alvo):
    entropia_total = 0
    total = len(dataset)
    for valor in dataset[atributo].unique():
        subset = dataset[dataset[atributo] == valor]
        if not subset.empty:
            proporcao = len(subset) / total
            entropia_subset = calcular_entropia(subset[coluna_alvo])
            entropia_total += proporcao * entropia_subset
    return entropia_total

def calcular_ganho(dataset, atributo, coluna_alvo):
    return calcular_entropia(dataset[coluna_alvo]) - entropia_condicional(dataset, atributo, coluna_alvo)

# Classe para os nós da árvore (sem alteração)
class NoArvoreID3:
    def __init__(self, atributo=None, filhos=None, classe=None):
        self.atributo = atributo
        self.filhos = filhos or {}
        self.classe = classe

# Classe principal da Árvore com a função de previsão adicionada
class ArvoreID3:
    def __init__(self):
        self.raiz = None
        self.classe_majoritaria_geral = None  # ADICIONADO: Para fallback na previsão

    def treinar(self, dataset, atributos, coluna_alvo):
        """Constrói a árvore e armazena a raiz e a classe majoritária."""
        # ADICIONADO: Armazena a classe mais comum para usar em previsões incertas
        self.classe_majoritaria_geral = dataset[coluna_alvo].mode()[0]
        self.raiz = self._construir_arvore(dataset, atributos, coluna_alvo)

    def _construir_arvore(self, dataset, atributos, coluna_alvo):
        # Caso base 1: todas as instâncias têm a mesma classe
        if len(dataset[coluna_alvo].unique()) == 1:
            return NoArvoreID3(classe=dataset[coluna_alvo].iloc[0])
        
        # Caso base 2: não há mais atributos para dividir
        if not atributos:
            return NoArvoreID3(classe=dataset[coluna_alvo].mode()[0])
        
        # Melhor atributo pelo ganho
        ganhos = {atributo: calcular_ganho(dataset, atributo, coluna_alvo) for atributo in atributos}
        melhor_atributo = max(ganhos, key=ganhos.get)
        
        no = NoArvoreID3(atributo=melhor_atributo)
        atributos_restantes = [a for a in atributos if a != melhor_atributo]

        # Itera sobre os valores únicos do melhor atributo para criar os ramos
        for valor in dataset[melhor_atributo].unique():
            subset = dataset[dataset[melhor_atributo] == valor]
            
            # Se um ramo não tiver mais exemplos, cria uma folha com a classe majoritária do pai
            if subset.empty:
                no.filhos[valor] = NoArvoreID3(classe=dataset[coluna_alvo].mode()[0])
            else:
                # Chama a construção da árvore recursivamente para o subconjunto
                no.filhos[valor] = self._construir_arvore(subset, atributos_restantes, coluna_alvo)
        return no

    # ---- INÍCIO DAS FUNÇÕES DE PREVISÃO ADICIONADAS ----
    
    def predict(self, X_teste):
        """
        Recebe um DataFrame de teste e retorna uma lista de previsões.
        """
        if self.raiz is None:
            raise ValueError("O modelo precisa ser treinado com o método .treinar() antes de fazer previsões.")
        
        # Usa o método .apply para percorrer cada linha do DataFrame e fazer a previsão
        return X_teste.apply(self._prever_instancia, axis=1, args=(self.raiz,)).tolist()

    def _prever_instancia(self, instancia, no_atual):
        """
        Navega na árvore recursivamente para classificar uma única instância (linha).
        """
        # Caso base: se chegamos a um nó folha, retornamos sua classe
        if no_atual.classe is not None:
            return no_atual.classe
        
        # Obtém o valor do atributo de decisão para a instância atual
        valor_da_instancia = instancia.get(no_atual.atributo)
        
        # Procura o ramo correspondente ao valor da instância
        if valor_da_instancia in no_atual.filhos:
            # Continua a busca recursivamente no filho correspondente
            return self._prever_instancia(instancia, no_atual.filhos[valor_da_instancia])
        else:
            # Fallback: se o valor não foi visto no treino, retorna a classe majoritária geral
            return self.classe_majoritaria_geral

    # ---- FIM DAS FUNÇÕES DE PREVISÃO ADICIONADAS ----

# COPIE E COLE ESTES DOIS MÉTODOS DENTRO DA SUA CLASSE ArvoreID3
# (Substituindo os métodos de impressão antigos)

    def imprimir_arvore(self):
        """Método público para iniciar a impressão da árvore a partir da raiz."""
        if self.raiz is None:
            print("A árvore não foi treinada.")
        else:
            # Inicia a chamada recursiva
            self._imprimir_no(self.raiz)

    def _imprimir_no(self, no: NoArvoreID3, indent: str = ""):
        """Imprime recursivamente a estrutura da árvore ID3 (multi-galhos)."""
        
        # CASO BASE: Se for um nó folha, imprime a previsão e para a recursão.
        if no.classe is not None:
            predicao = "Sobreviveu" if no.classe == 1 else "Não Sobreviveu"
            print(f"{indent}--> PREDIÇÃO: {predicao} (classe={no.classe})")
            return

        # NÓ DE DECISÃO: Imprime o atributo que está sendo usado para dividir.
        print(f"{indent}NÓ: Divisão por '{no.atributo}'")
        
        # Itera sobre cada possível valor do atributo (as chaves do dicionário de filhos).
        # Esta é a principal diferença em relação ao CART.
        for valor, filho in no.filhos.items():
            # Imprime a regra para seguir este galho específico.
            print(f"{indent}  - Se '{no.atributo}' == '{valor}':")
            
            # Faz a chamada recursiva para o filho, aumentando a indentação.
            self._imprimir_no(filho, indent + "    ")