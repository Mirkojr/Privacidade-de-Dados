import os
import numpy as np
import pandas as pd
import kagglehub
from typing import Tuple, Dict

# Importando as classes e funções dos outros módulos
from LaplaceKnn import LaplaceKNN
from KnnTradicional import knn_tradicional_k10
from metrics import distancia_euclidiana

# --- CONFIGURAÇÃO ---
CONFIG = {
    "REPO_URL": "wenruliu/adult-income-dataset",
    "FILENAME": "adult.csv",
    "TARGET_COL": "income",
    "DROP_COLS": ['fnlwgt', 'education', 'capital-gain', 'capital-loss', 'hours-per-week'],
    "IGNORE_ENCODE": ['age', 'educational-num'],
    "TRAIN_SPLIT": 0.7,
    "SEED": 42,
    "EPSILONS": [0.1, 0.5, 1.0, 5.0],
    "RAIO": 6
}

def carregar_dataset() -> pd.DataFrame:
    """Baixa e carrega o dataset."""
    try:
        path = kagglehub.dataset_download(CONFIG["REPO_URL"])
        file_path = os.path.join(path, CONFIG["FILENAME"])
        return pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar dataset: {e}")

def preprocessar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa o dataset."""
    df = df.drop(columns=CONFIG["DROP_COLS"], errors='ignore')
    df = df.replace('?', np.nan).dropna()
    return df

def codificar_dados(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Codifica colunas categóricas."""
    df_enc = df.copy()
    
    # Seleciona colunas categóricas ignorando as especificadas  
    cols = [c for c in df_enc.columns if c not in CONFIG["IGNORE_ENCODE"] and df_enc[c].dtype == 'object']
    
    mapeamento = {}
    for c in cols:
        unique_vals = df_enc[c].unique()
        mapa = {val: i for i, val in enumerate(unique_vals)}
        mapeamento[c] = mapa
        df_enc[c] = df_enc[c].map(mapa)
        
    return mapeamento, df_enc

def obter_treino_teste(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """Divide em treino e teste e converte para NumPy."""
    train_df = df.sample(frac=CONFIG["TRAIN_SPLIT"], random_state=CONFIG["SEED"])
    test_df = df.drop(train_df.index)
    
    # Função auxiliar interna para separar X e y
    def split_xy(d):
        x = d.drop(columns=[CONFIG["TARGET_COL"]]).values
        y = d[CONFIG["TARGET_COL"]].values
        return x, y

    X_tr, y_tr = split_xy(train_df)
    X_te, y_te = split_xy(test_df)
    
    return X_tr, y_tr, X_te, y_te

def salvar_resultados(filename: str, conteudo: str):
    with open(filename, "w") as f:
        f.write(conteudo)

# --- EXECUÇÃO ---
if __name__ == "__main__":
    print(">>> Iniciando Pipeline...")
    
    # 1. Dados
    raw_data = carregar_dataset()
    clean_data = preprocessar(raw_data)
    _, encoded_data = codificar_dados(clean_data)
    X_train, y_train, X_test, y_test = obter_treino_teste(encoded_data)
    
    classes_unicas = np.unique(y_train)
    
    # 2. Tradicional
    print("\n>>> Executando KNN Tradicional...")
    preds_trad = knn_tradicional_k10(X_train, y_train, X_test, k=10)
    acc_trad = np.mean(preds_trad == y_test)
    print(f"Acurácia Tradicional: {acc_trad:.4f}")
    
    salvar_resultados("resultado_tradicional.txt", 
                      f"Acuracia: {acc_trad}\nPredicoes: {preds_trad.tolist()}")

    # 3. Privado
    print("\n>>> Executando KNN Privado...")
    knn_privado = LaplaceKNN(X_train, y_train, 0, radius_r=CONFIG["RAIO"])
    
    # Estratégia de Fallback (Classe majoritária global) para casos sem vizinhos
    classe_fallback = np.argmax(np.bincount(y_train))

    for eps in CONFIG["EPSILONS"]:
        print(f"   -> Rodando Epsilon: {eps}")
        knn_privado.atualizar_epsilon(eps)
        
        preds_priv = []
        for amostra in X_test:
 
            contagens = knn_privado.calcular_contagens_ruidosas(amostra, classes_unicas)
            
            if contagens:
                pred = knn_privado.decidir_vencedor_ruidoso(contagens)
            else:
                pred = classe_fallback
            preds_priv.append(pred)

        preds_priv = np.array(preds_priv)
        acc_priv = np.mean(preds_priv == y_test)
        print(f"      Acurácia (eps={eps}): {acc_priv:.4f}")
        
        salvar_resultados(f"resultado_privado_eps_{eps}.txt",
                          f"Epsilon: {eps}\nRaio: {CONFIG['RAIO']}\nAcuracia: {acc_priv}\nPredicoes: {preds_priv.tolist()}")
    
    print("\n>>> Processo Finalizado.")