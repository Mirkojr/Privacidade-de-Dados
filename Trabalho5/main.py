import kagglehub
import pandas as pd
from LaplaceKnn import LaplaceKNN
from KnnTradicional import knn_tradicional_k10
import numpy as np
import os

def data_download():
    """ Realiza o download do dataset 'adult.csv' do Kaggle.
        Retorna o DataFrame com os dados.
    """
    path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
    csv_file = "adult.csv"
    data = pd.read_csv(os.path.join(path, csv_file))
    return data

def preprocess_data(data: pd.DataFrame):
    """ Realiza o pré-processamento dos dados conforme especificação.
        Remove colunas irrelevantes, trata valores faltantes.
    """
    cols_to_drop = ['fnlwgt', 'education', 'capital-gain', 'capital-loss', 'hours-per-week']
    data = data.drop(columns=cols_to_drop)
    
    data = data.replace('?', np.nan)
    data = data.dropna()
    return data

def label_encoder(data: pd.DataFrame):
    """ Codifica as colunas categóricas em números inteiros.
        Retorna o dicionário de mapeamento e o DataFrame codificado.
    """
    df_encoded = data.copy()

    # Seleciona as colunas categóricas (excluindo 'age' e 'educational-num')
    cols = df_encoded.drop(columns=['age', 'educational-num']).columns

    mapeamento_completo = {}
    for coluna in cols:
        # mapeamento de cada valor único da coluna para um número inteiro 
        mapa = {label: i for i, label in enumerate(df_encoded[coluna].unique())}
        # guarda o mapeamento que é um dicionário
        mapeamento_completo[coluna] = mapa
        # substitui a coluna codificada para o DataFrame 
        df_encoded[coluna] = df_encoded[coluna].map(mapa)
    
    return mapeamento_completo, df_encoded

def dividir_dados_treino_teste(data, target_col='income'):
    """ Divide o DataFrame em conjuntos de treino e teste (70%/30%).
        Retorna X_train, y_train, X_test, y_test como arrays NumPy.
    """
    train_df = data.sample(frac=0.7, random_state=42)
    
    # O TESTE é o dataset original MENOS as linhas que foram pro treino
    test_df = data.drop(train_df.index)
    
    # Separa X e y e converte para array do NumPy com .values (necessário para o KNN)
    # Para treino
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    # Para teste
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    
    return X_train, y_train, X_test, y_test

def print_mapeamento(mapeamentoDict: dict):
    """ Imprime o mapeamento das colunas categóricas para seus valores inteiros.
    """
    for key, value in mapeamentoDict.items():
        print(f"{key}:")
        for item in value.items():
            print(f"  {item}")

def qtd_classes_Mapeamento(mapeamentoDict: dict):
    """ Retorna a quantidade total de classes no mapeamento.
    """
    qtd = 0
    ultima_chave = list(mapeamentoDict.keys())[-1] 
    print("O somatório das classes é: ", end="")
    for key, value in mapeamentoDict.items():
        print(len(value), end="=" if ultima_chave == key else "+")
        qtd += len(value)

    return qtd

def classe_Majoritaria(array):
    """ Retorna a classe majoritária do conjunto dado.
    """
    valores, contagens = np.unique(array, return_counts=True)
    indice_maior = np.argmax(contagens)
    classe_majoritaria = valores[indice_maior]
    return classe_majoritaria

def preparar_dados():
    # DOWNLOAD DO ARQUIVO 'adult.csv'
    try:
        data = data_download()
    except:
        print("Erro no download. Verifique internet/caminho.")
        return None, None, None, None

    # PREPROCESSAMENTO
    data_preprocessed = preprocess_data(data)

    # CODIFICAÇÃO DE LABELS
    dict_labels, df_labels_encoded = label_encoder(data_preprocessed)

    # DIVISÃO TREINO/TESTE
    X_train, y_train, X_test, y_test = dividir_dados_treino_teste(df_labels_encoded, target_col='income')

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    
    print("--- Preparando Dados ---")
    X_train, y_train, X_test, y_test = preparar_dados()
    
    # Definir labelsSet
    labels_set = np.unique(y_train) 
    
    # --- KNN TRADICIONAL (K=10) ---
    preds_trad = knn_tradicional_k10(X_train, y_train, X_test, k=10)
    acc_trad = np.mean(preds_trad == y_test)
    print(f"Acurácia KNN Tradicional: {acc_trad:.4f}")
    
    # Salvar arquivo tradicional
    with open("resultado_tradicional.txt", "w") as f:
        f.write(f"Acuracia: {acc_trad}\nPredicoes: {preds_trad.tolist()}")

    # --- KNN PRIVADO  ---
    valores_epsilon = [0.1, 0.5, 1.0, 5.0]
    raio = 6 
    
    # Inicializa o classificador LaplaceKNN
    knn_privado = LaplaceKNN(
        dataset=X_train,
        labels=y_train,
        privacy_budget=0,
        radius_r=raio
    )

    # Obtém a classe majoritária para uso em casos sem vizinhos
    classe_majoritaria = classe_Majoritaria(y_train)
    # Loop pelos valores de epsilon
    for eps in valores_epsilon:
        knn_privado.atualizar_epsilon(eps)

        print(f"\n--- Rodando Privado | Epsilon: {eps} | Raio: {raio} ---")
        
        predicoes_privadas = []
        # Loop pelos dados de teste
        for i, amostra in enumerate(X_test):
            if i % 100 == 0: print(f"Progresso Eps {eps}: {i}/{len(X_test)}")
            
            # Obter contagens ruidosas para a amostra
            contagens = knn_privado.calcular_contagens_ruidosas(amostra, labels_set)
            
            # Decidir o vencedor com base nas contagens ruidosas
            if contagens:
                vencedor = knn_privado.decidir_vencedor_ruidoso(contagens)
                predicoes_privadas.append(vencedor)
            else:
                # Se não tem vizinho no raio 6, chutamos a classe majoritária (0) ou -1
                predicoes_privadas.append(classe_majoritaria) 

        # Calcula acurácia 
        predicoes_privadas = np.array(predicoes_privadas)
        acc_priv = np.mean(predicoes_privadas == y_test)
        print(f"Acurácia Privada (eps={eps}): {acc_priv:.4f}")

        # Salvar arquivo privado
        nome_arq = f"resultado_privado_eps_{eps}.txt"
        with open(nome_arq, "w") as f:
            f.write(f"Epsilon: {eps}\nRaio: {raio}\nAcuracia: {acc_priv}\nPredicoes: {predicoes_privadas.tolist()}")

    print("\nFim do processamento.")