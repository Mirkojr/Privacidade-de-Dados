import os
import numpy as np
import pandas as pd
import kagglehub
from typing import Tuple, Dict
import matplotlib.pyplot as plt 

# Imports das classes
from ExponentialKNN import ExponentialKNN
from RnnTradicional import rnn_tradicional

# --- CONFIGURAÇÃO ---
CONFIG = {
    "REPO_URL": "wenruliu/adult-income-dataset",
    "FILENAME": "adult.csv",
    "TARGET_COL": "income",
    "DROP_COLS": ['fnlwgt', 'education', 'capital-gain', 'capital-loss', 'hours-per-week'],
    "TRAIN_SPLIT": 0.7,
    "SEED": 42,
    "EPSILONS": [0.1, 0.5, 1, 5, 10, 100], 
    "RADIUS": [1, 3, 6, 9],
    "GRAPH_FILE": "grafico_acuracia_exponencial.png",
    # Nomes dos arquivos
    "FILE_TRAD": "RELATORIO_TRADICIONAL.txt",
    "FILE_PRIV": "RELATORIO_EXPONENCIAL.txt"
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

    cols = ['workclass','marital-status','occupation','relationship','race','gender','native-country','income']

    mapeamento = {}
    for c in cols: 
        unique_vals = df_enc[c].unique()
        mapa = {val: i for i, val in enumerate(unique_vals)}
        mapeamento[c] = mapa
        df_enc[c] = df_enc[c].map(mapa)
        
    return mapeamento, df_enc

def obter_treino_teste(X:pd.DataFrame, y:pd.Series) -> Tuple[np.ndarray, ...]:
    """Divide em treino e teste e converte para NumPy."""
    X_treino = X.sample(frac=CONFIG["TRAIN_SPLIT"], random_state=CONFIG["SEED"])
    y_treino = y.loc[X_treino.index]

    X_teste = X.drop(X_treino.index)
    y_teste = y.loc[X_teste.index]
    
    return X_treino.values, y_treino.values, X_teste.values, y_teste.values


def salvar_resultados(filename: str, conteudo: str):
    with open(filename, "w") as f:
        f.write(conteudo)

def gerar_grafico_comparativo(resultados_privados, resultados_tradicionais):
    plt.figure(figsize=(12, 7))
    cores = ['b', 'g', 'r', 'c']
    epsilons = CONFIG["EPSILONS"]
    str_epsilons = [str(e) for e in epsilons]
    
    for i, r in enumerate(CONFIG["RADIUS"]):
        cor = cores[i % len(cores)]
        accs = resultados_privados[r]
        plt.plot(str_epsilons, accs, marker='o', linestyle='-', color=cor, label=f'Exp KNN (r={r})')
        acc_trad = resultados_tradicionais[r]
        plt.axhline(y=acc_trad, color=cor, linestyle='--', alpha=0.5, label=f'Tradicional (r={r}): {acc_trad:.2f}')

    plt.title('Acurácia: Mecanismo Exponencial vs Tradicional')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Acurácia')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(CONFIG["GRAPH_FILE"])
    print(f"\n[Gráfico] Salvo em: {CONFIG['GRAPH_FILE']}")

# --- MAIN ---
if __name__ == "__main__":
    print(">>> Iniciando Trabalho 6 - Mecanismo Exponencial...")
    
    # Preparação dos Dados
    raw_data = carregar_dataset()
    clean_data = preprocessar(raw_data)
    _, encoded_data = codificar_dados(clean_data)

    X = encoded_data.drop(columns=[CONFIG["TARGET_COL"]])
    y = encoded_data[CONFIG["TARGET_COL"]]

    X_train, y_train, X_test, y_test = obter_treino_teste(X, y)
    classes_unicas = np.unique(y_train)
    
    print(f"Tamanho Treino: {len(X_train)} | Tamanho Teste: {len(X_test)}")
    
    hist_acc_privado = {r: [] for r in CONFIG["RADIUS"]}
    hist_acc_tradicional = {} 

    knn_exp = ExponentialKNN(X_train, y_train, privacy_budget_total=1.0, radius_r=1)

    # --- BUFFERS PARA GUARDAR OS RESULTADOS ---
    buffer_tradicional = "==== RELATORIO R-NN TRADICIONAL ====\n\n"
    buffer_privado = "==== RELATORIO R-NN EXPONENCIAL (DP) ====\n\n"

    # Loop Principal
    for r in CONFIG["RADIUS"]:
        print(f"\n==== Processando Raio r = {r} ====")
        
        # A) Rodar Tradicional
        preds_trad = rnn_tradicional(X_train, y_train, X_test, radius=r)
        acc_trad = np.mean(preds_trad == y_test)
        hist_acc_tradicional[r] = acc_trad
        print(f"   [Tradicional] Acurácia: {acc_trad:.4f}")
        
        # Acumula no buffer tradicional
        buffer_tradicional += f"--- RAIO: {r} ---\n"
        buffer_tradicional += f"Acuracia: {acc_trad:.4f}\n"
        buffer_tradicional += f"Predicoes (primeiras 50): {preds_trad[:50].tolist()} ... [truncado]\n"
        buffer_tradicional += "="*40 + "\n\n"

        # B) Rodar Privado
        for eps in CONFIG["EPSILONS"]:
            knn_exp.atualizar_params(epsilon_total=eps, radius_r=r)
            preds_priv = knn_exp.predict(X_test, classes_unicas)
            acc_priv = np.mean(preds_priv == y_test)
            hist_acc_privado[r].append(acc_priv)
            
            print(f"   [Exponencial] eps={eps}: Acurácia={acc_priv:.4f}")
            
            # Acumula no buffer privado
            buffer_privado += f"--- RAIO: {r} | EPSILON: {eps} ---\n"
            buffer_privado += f"Acuracia: {acc_priv:.4f}\n"
            buffer_privado += f"Predicoes (primeiras 50): {preds_priv[:50].tolist()} ... [truncado]\n"
            buffer_privado += "-"*30 + "\n\n"

    # - Salvar os arquivos consolidados
    print("\n>>> Salvando relatórios...")
    salvar_resultados(CONFIG["FILE_TRAD"], buffer_tradicional)
    salvar_resultados(CONFIG["FILE_PRIV"], buffer_privado)
    print(f"   -> {CONFIG['FILE_TRAD']} salvo com sucesso.")
    print(f"   -> {CONFIG['FILE_PRIV']} salvo com sucesso.")

    # - Gerar Gráfico
    gerar_grafico_comparativo(hist_acc_privado, hist_acc_tradicional)
    print("\n>>> Fim da Execução.")