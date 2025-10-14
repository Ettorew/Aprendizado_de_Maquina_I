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
    Calcula acurácia, precisão, recall e F1 de forma genérica para qualquer rótulo binário.
    Retorna string formatada.
    """
    assert len(y_true) == len(y_pred), "Listas de tamanhos diferentes"
    
    classes = set(y_true)
    if len(classes) != 2:
        raise ValueError("Essa versão assume classificação binária")
    
    c1, c2 = classes
    # Vamos considerar c1 como 'positiva' e c2 como 'negativa'
    tp = sum((yt == c1 and yp == c1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == c1 and yp == c2) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == c2 and yp == c1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == c2 and yp == c2) for yt, yp in zip(y_true, y_pred))

    acuracia = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

    return f"Acurácia: {acuracia:.2f}, Precisão: {precisao:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
