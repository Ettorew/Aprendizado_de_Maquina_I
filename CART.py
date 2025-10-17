# ==========================================================
# ALGORITMO CART CORRIGIDO E ROBUSTO
# ==========================================================
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class NoArvoreCART:
    def __init__(self, atributo: Optional[str] = None, limite: Any = None, 
                 filho_esq: Optional['NoArvoreCART'] = None, filho_dir: Optional['NoArvoreCART'] = None, 
                 classe: Optional[Any] = None):
        self.atributo = atributo  # Atributo usado para a divisão
        self.limite = limite      # Limite (numérico) ou valor (categórico) para a divisão
        self.filho_esq = filho_esq
        self.filho_dir = filho_dir
        self.classe = classe      # Valor da previsão (apenas para nós folha)

class ArvoreCART:
    def __init__(self, max_depth: Optional[int] = 10, min_samples_leaf: int = 1):
        self.raiz = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    # --- Métodos de Cálculo do Gini (sem alterações) ---
    @staticmethod
    def _indice_gini(coluna_alvo: pd.Series) -> float:
        if coluna_alvo.empty: return 0
        proporcoes = coluna_alvo.value_counts(normalize=True)
        return 1 - sum(proporcoes ** 2)

    def _ganho_gini_ponderado(self, subsets: List[pd.Series], total_amostras: int) -> float:
        gini_ponderado = 0
        for subset in subsets:
            if not subset.empty:
                gini_ponderado += (len(subset) / total_amostras) * self._indice_gini(subset)
        return gini_ponderado

    # --- Métodos para Encontrar o Melhor Split ---
    def _melhor_split_numerico(self, dataset: pd.DataFrame, atributo: str, coluna_alvo: str) -> Tuple[Optional[float], float]:
        melhor_limite, melhor_score = None, float('inf')
        
        valores_unicos = sorted(dataset[atributo].unique())
        candidatos = [(valores_unicos[i] + valores_unicos[i-1]) / 2 for i in range(1, len(valores_unicos))]
        
        for limite in candidatos:
            subset_esq = dataset[dataset[atributo] <= limite]
            subset_dir = dataset[dataset[atributo] > limite]
            
            if len(subset_esq) < self.min_samples_leaf or len(subset_dir) < self.min_samples_leaf:
                continue
                
            gini_atual = self._ganho_gini_ponderado([subset_esq[coluna_alvo], subset_dir[coluna_alvo]], len(dataset))
            if gini_atual < melhor_score:
                melhor_score, melhor_limite = gini_atual, limite
                
        return melhor_limite, melhor_score

    # CORREÇÃO: Lógica para split binário em atributos categóricos
    def _melhor_split_categorico(self, dataset: pd.DataFrame, atributo: str, coluna_alvo: str) -> Tuple[Optional[Any], float]:
        melhor_valor, melhor_score = None, float('inf')

        for valor in dataset[atributo].unique():
            subset_esq = dataset[dataset[atributo] == valor]
            subset_dir = dataset[dataset[atributo] != valor]

            if len(subset_esq) < self.min_samples_leaf or len(subset_dir) < self.min_samples_leaf:
                continue
            
            gini_atual = self._ganho_gini_ponderado([subset_esq[coluna_alvo], subset_dir[coluna_alvo]], len(dataset))
            if gini_atual < melhor_score:
                melhor_score, melhor_valor = gini_atual, valor
                
        return melhor_valor, melhor_score

    # --- Método Principal de Treinamento ---
    def treinar(self, X: pd.DataFrame, y: pd.Series):
        dataset = pd.concat([X, y], axis=1)
        self.raiz = self._construir_arvore(dataset, y.name, 0)

    def _construir_arvore(self, dataset: pd.DataFrame, coluna_alvo: str, depth: int) -> NoArvoreCART:
        # --- Critérios de Parada (criação de nó folha) ---
        classe_majoritaria = dataset[coluna_alvo].mode()[0]
        if (len(dataset[coluna_alvo].unique()) == 1 or
            len(dataset) < self.min_samples_leaf * 2 or
            (self.max_depth is not None and depth >= self.max_depth)):
            return NoArvoreCART(classe=classe_majoritaria)

        # --- Encontrar o melhor split possível em todos os atributos ---
        melhor_split = {'score': float('inf')}
        atributos = list(dataset.columns)
        atributos.remove(coluna_alvo)

        for atributo in atributos:
            if pd.api.types.is_numeric_dtype(dataset[atributo]):
                limite, score = self._melhor_split_numerico(dataset, atributo, coluna_alvo)
            else: # Categórico
                limite, score = self._melhor_split_categorico(dataset, atributo, coluna_alvo)
            
            if score < melhor_split['score']:
                melhor_split = {'atributo': atributo, 'limite': limite, 'score': score}
        
        # Se nenhum split válido foi encontrado, cria uma folha
        if melhor_split['score'] == float('inf'):
            return NoArvoreCART(classe=classe_majoritaria)

        # --- Dividir o dataset com base no melhor split encontrado ---
        attr, limit = melhor_split['atributo'], melhor_split['limite']
        if pd.api.types.is_numeric_dtype(dataset[attr]):
            subset_esq = dataset[dataset[attr] <= limit]
            subset_dir = dataset[dataset[attr] > limit]
        else: # Categórico
            subset_esq = dataset[dataset[attr] == limit]
            subset_dir = dataset[dataset[attr] != limit]
        
        # --- Construir os filhos recursivamente ---
        filho_esq = self._construir_arvore(subset_esq, coluna_alvo, depth + 1)
        filho_dir = self._construir_arvore(subset_dir, coluna_alvo, depth + 1)
        
        return NoArvoreCART(atributo=attr, limite=limit, filho_esq=filho_esq, filho_dir=filho_dir)

    # ==========================================================
    # NOVA FUNÇÃO DE PREVISÃO PARA O CART
    # ==========================================================
    def prever(self, X: pd.DataFrame) -> List[Any]:
        """Faz previsões para um DataFrame de teste."""
        if self.raiz is None:
            raise ValueError("A árvore não foi treinada. Chame o método .treinar() primeiro.")
        return [self._prever_amostra(self.raiz, amostra) for _, amostra in X.iterrows()]

    def _prever_amostra(self, no: NoArvoreCART, amostra: pd.Series) -> Any:
        """Navega recursivamente na árvore para classificar uma única amostra."""
        # Se o nó for uma folha, retorna sua classe
        if no.classe is not None:
            return no.classe
        
        valor_amostra = amostra.get(no.atributo)
        
        # Se o atributo da árvore não existir na amostra, não podemos decidir
        if valor_amostra is None:
            # Em um cenário real, precisaríamos de uma estratégia de fallback.
            # Aqui, para simplificar, vamos descer pelo galho maior (pode ser aprimorado)
            return self._prever_amostra(no.filho_esq, amostra) # Decisão arbitrária

        # Verifica se o atributo é numérico ou categórico pelo tipo do limite
        if isinstance(no.limite, (int, float)): # Divisão Numérica
            if valor_amostra <= no.limite:
                return self._prever_amostra(no.filho_esq, amostra)
            else:
                return self._prever_amostra(no.filho_dir, amostra)
        else: # Divisão Categórica
            if valor_amostra == no.limite:
                return self._prever_amostra(no.filho_esq, amostra)
            else:
                return self._prever_amostra(no.filho_dir, amostra)

    # ----------------------------
    # PODA (MANTIDO)
    # ----------------------------
    def poda(self, dataset, coluna_alvo, alpha=0.0):
        # A poda deve ser feita no dataset de VALIDAÇÃO, não de treino.
        # Estamos mantendo o 'treino' conforme seu código, mas idealmente seria 'validacao'.
        self.raiz = self._poda_subarvore(self.raiz, dataset, coluna_alvo, alpha)

    def _poda_subarvore(self, no, dataset, coluna_alvo, alpha):
        # O nó agora pode ter no.classe mesmo sendo interno, mas o teste principal da poda
        # é se ele tem filhos (se não for nó terminal por CCP, ele tem filhos).
        if no.atributo is None: # Se for uma folha por construção, não poda
            return no
        
        # Poda recursiva nos filhos (só entra se o nó for interno)
        # O código de poda aqui assume que no.classe é o modo do nó pai
        for valor, filho in list(no.filhos.items()):
            # ... (seu código de poda recursiva)
            if no.limite is not None:
                subset = dataset[dataset[no.atributo] <= no.limite] if valor == "<=" else dataset[dataset[no.atributo] > no.limite]
            else:
                subset = dataset[dataset[no.atributo] == valor]
            # Chamada recursiva para poda do filho
            no.filhos[valor] = self._poda_subarvore(filho, subset, coluna_alvo, alpha)
        
        # Reavaliação para a Poda CCP no nó atual
        erro_total = self._erro_subarvore(no, dataset, coluna_alvo)
        classe_mais_comum = dataset[coluna_alvo].mode()[0]
        erro_no = sum(dataset[coluna_alvo] != classe_mais_comum)

        n_folhas = self._conta_folhas(no)
        
        # A PODa AQUI: compara o custo-complexidade da subárvore com o nó folha
        if n_folhas > 1 and (erro_no + alpha) <= (erro_total + alpha * n_folhas):
            # PODAR: Retorna o nó como folha. O atributo se torna None.
            return NoArvoreCART(classe=classe_mais_comum)
        
        # Se não podar, atualiza a classe de fallback do nó interno (opcional)
        no.classe = classe_mais_comum
        return no

    # COPIE E COLE ESTES DOIS MÉTODOS DENTRO DA SUA CLASSE ArvoreCART em CART.py

    def imprimir_arvore(self):
        """Método público para iniciar a impressão da árvore a partir da raiz."""
        if self.raiz is None:
            print("A árvore não foi treinada.")
        else:
            self._imprimir_no(self.raiz)

    def _imprimir_no(self, no: NoArvoreCART, indent: str = ""):
        """Imprime recursivamente a estrutura da árvore."""
        # Se for um nó folha, imprime a previsão
        if no.classe is not None:
            predicao = "Sobreviveu" if no.classe == 1 else "Não Sobreviveu"
            print(f"{indent}--> PREDIÇÃO: {predicao} (classe={no.classe})")
            return

        # Se for um nó de decisão, imprime a regra
        if isinstance(no.limite, (int, float)): # Regra Numérica
            regra = f"Se '{no.atributo}' <= {no.limite:.3f}"
        else: # Regra Categórica
            regra = f"Se '{no.atributo}' == '{no.limite}'"
        
        print(f"{indent}{regra}:")
        
        # Imprime o filho da esquerda (condição verdadeira)
        self._imprimir_no(no.filho_esq, indent + "  |-- V: ")
        
        # Imprime o filho da direita (condição falsa)
        print(f"{indent}Senão:")
        self._imprimir_no(no.filho_dir, indent + "  |-- F: ")