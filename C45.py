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

# Classe para os nós
class NoArvoreC45:
    def __init__(self, atributo=None, limite=None, filhos=None, classe=None):
        self.atributo = atributo
        self.limite = limite
        self.filhos = filhos or {}
        self.classe = classe

class ArvoreC45:
    def __init__(self):
        self.raiz = None
        self.classe_majoritaria_geral = None  # ADICIONADO: Para fallback na previsão

    def treinar(self, dataset, atributos, coluna_alvo):
        """Constrói a árvore e armazena a raiz e a classe majoritária."""
        # ADICIONADO: Armazena a classe mais comum para usar em previsões incertas
        self.classe_majoritaria_geral = dataset[coluna_alvo].mode()[0]
        self.raiz = self._construir_arvore(dataset, atributos, coluna_alvo)
        
    def podar(self, dataset_poda, coluna_alvo):
        """Aplica a poda na árvore já construída."""
        if self.raiz:
            self.raiz = self._poda_c45(self.raiz, dataset_poda, coluna_alvo)

    # (Seu método construir_arvore, com pequenas melhorias de robustez)
    def _construir_arvore(self, dataset, atributos, coluna_alvo):
        if dataset.empty:
            return NoArvoreC45(classe=self.classe_majoritaria_geral)
        if len(dataset[coluna_alvo].unique()) == 1:
            return NoArvoreC45(classe=dataset[coluna_alvo].iloc[0])
        if not atributos:
            return NoArvoreC45(classe=dataset[coluna_alvo].mode()[0])
        
        melhor_score = -float("inf")
        melhor_atributo, melhor_limite = None, None

        for atributo in atributos:
            if pd.api.types.is_numeric_dtype(dataset[atributo]):
                limite, score = melhor_limite_numerico(dataset, atributo, coluna_alvo, usar_razao=True)
            else:
                score = calcular_razao_ganho(dataset, atributo, coluna_alvo)
                limite = None
            if score > melhor_score:
                melhor_score = score
                melhor_atributo = atributo
                melhor_limite = limite
        
        if melhor_atributo is None:
            return NoArvoreC45(classe=dataset[coluna_alvo].mode()[0])

        no = NoArvoreC45(atributo=melhor_atributo, limite=melhor_limite)
        atributos_restantes = [a for a in atributos if a != melhor_atributo]

        if melhor_limite is not None:
            subset_esq = dataset[dataset[melhor_atributo] <= melhor_limite]
            subset_dir = dataset[dataset[melhor_atributo] > melhor_limite]
            no.filhos[f"<= {melhor_limite:.3f}"] = self._construir_arvore(subset_esq, atributos_restantes, coluna_alvo)
            no.filhos[f"> {melhor_limite:.3f}"] = self._construir_arvore(subset_dir, atributos_restantes, coluna_alvo)
        else:
            for valor in dataset[melhor_atributo].unique():
                subset = dataset[dataset[melhor_atributo] == valor]
                no.filhos[valor] = self._construir_arvore(subset, atributos_restantes, coluna_alvo)
        return no

    def predict(self, X_teste):
        """
        Recebe um DataFrame de teste e retorna uma lista de previsões.
        """
        if self.raiz is None:
            raise ValueError("O modelo precisa ser treinado com .treinar() antes de prever.")
        return X_teste.apply(self._prever_instancia, axis=1, args=(self.raiz,)).tolist()

    def _prever_instancia(self, instancia, no_atual):
        """
        Navega na árvore recursivamente para classificar uma única instância.
        """
        # Caso base: se é um nó folha, retorna a classe
        if no_atual.classe is not None:
            return no_atual.classe

        atributo_do_no = no_atual.atributo
        valor_da_instancia = instancia.get(atributo_do_no)

        # Se o valor for nulo na instância de teste, não podemos prosseguir
        if pd.isna(valor_da_instancia):
            return self.classe_majoritaria_geral

        # --- Lógica Principal: Decide entre Divisão Numérica ou Categórica ---
        
        # CASO 1: NÓ DE DIVISÃO NUMÉRICA (possui um 'limite')
        if no_atual.limite is not None:
            if valor_da_instancia <= no_atual.limite:
                # Procura a chave do filho que representa a condição '<='
                for chave_filho in no_atual.filhos:
                    if chave_filho.startswith('<='):
                        return self._prever_instancia(instancia, no_atual.filhos[chave_filho])
            else:
                # Procura a chave do filho que representa a condição '>'
                for chave_filho in no_atual.filhos:
                    if chave_filho.startswith('>'):
                        return self._prever_instancia(instancia, no_atual.filhos[chave_filho])
        
        # CASO 2: NÓ DE DIVISÃO CATEGÓRICA
        else:
            # Procura o ramo correspondente ao valor da instância
            if valor_da_instancia in no_atual.filhos:
                return self._prever_instancia(instancia, no_atual.filhos[valor_da_instancia])
        
        # FALLBACK: Se nenhum caminho for encontrado (ex: valor categórico não visto no treino)
        return self.classe_majoritaria_geral
    
    def imprimir_arvore(self):
        """Método público para iniciar a impressão da árvore a partir da raiz."""
        if self.raiz is None:
            print("A árvore não foi treinada.")
        else:
            # Inicia a chamada recursiva
            self._imprimir_no(self.raiz)

    def _imprimir_no(self, no: NoArvoreC45, indent: str = ""):
        """Imprime recursivamente a estrutura da árvore C4.5 (híbrida)."""
        
        # CASO BASE: Se for um nó folha, imprime a previsão e para.
        if no.classe is not None:
            predicao = "Sobreviveu" if no.classe == 1 else "Não Sobreviveu"
            print(f"{indent}--> PREDIÇÃO: {predicao} (classe={no.classe})")
            return

        # --- NÓ DE DECISÃO: LÓGICA HÍBRIDA ---
        
        # CASO 1: A divisão é NUMÉRICA (o nó possui um 'limite')
        if no.limite is not None:
            print(f"{indent}NÓ: Divisão Numérica por '{no.atributo}'")
            
            # Itera sobre os dois filhos (<= e >)
            for condicao, filho in no.filhos.items():
                print(f"{indent}  - Se valor {condicao}:")
                self._imprimir_no(filho, indent + "    ")

        # CASO 2: A divisão é CATEGÓRICA (semelhante ao ID3)
        else:
            print(f"{indent}NÓ: Divisão Categórica por '{no.atributo}'")
            
            # Itera sobre todos os valores/galhos possíveis
            for valor, filho in no.filhos.items():
                print(f"{indent}  - Se '{no.atributo}' == '{valor}':")
                self._imprimir_no(filho, indent + "    ")

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
