import os
import numpy as np
import pandas as pd
import kagglehub
from typing import Tuple, Dict, List # Para tipagem 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
# Importando as classes e funções dos outros módulos
from LaplaceKnn import LaplaceKNN
from KnnTradicional import knn_tradicional_k10


# --- CONFIGURAÇÃO ---
CONFIG = {
    "REPO_URL": "wenruliu/adult-income-dataset",
    "FILENAME": "adult.csv",
    "TARGET_COL": "income",
    "DROP_COLS": ['fnlwgt', 'education', 'capital-gain', 'capital-loss', 'hours-per-week'],
    "TRAIN_SPLIT": 0.7,
    "SEED": 42,
    "EPSILONS": [0.001, 0.5, 1, 5, 10],
    "RAIO": 6,
    "GRAPH_FILE": "grafico_acuracia.png"
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
        
def gerar_grafico_comparativo(epsilons: List[float], priv_accs: List[float], trad_acc: float):
    """Gera e salva o gráfico comparativo de acurácia."""
    plt.figure(figsize=(10, 6))
    
    # 1. Linha do KNN Privado
    plt.plot(epsilons, priv_accs, marker='o', linestyle='-', color='b', label='KNN Privado (Laplace)')
    
    # 2. Linha de Referência do KNN Tradicional
    # axhline cria uma linha horizontal em todo o eixo
    plt.axhline(y=trad_acc, color='r', linestyle='--', label=f'KNN Tradicional (k=10): {trad_acc:.4f}')
    
    # Configurações visuais
    plt.title(f'Impacto do Epsilon na Acurácia (Raio={CONFIG["RAIO"]})')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Acurácia')
    plt.xticks(epsilons, rotation=45) # Garante que os epsilons apareçam no eixo X
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Salvar
    plt.savefig(CONFIG["GRAPH_FILE"])
    print(f"\n[Gráfico] Salvo em: {CONFIG['GRAPH_FILE']}")
    plt.close()

# --- EXECUÇÃO ---
if __name__ == "__main__":
    print(">>> Iniciando Pipeline...")
    
    # 1. Dados
    raw_data = carregar_dataset()
    clean_data = preprocessar(raw_data)
    _, encoded_data = codificar_dados(clean_data)

    X = encoded_data.drop(columns=[CONFIG["TARGET_COL"]])
    y = encoded_data[CONFIG["TARGET_COL"]]

    X_train, y_train, X_test, y_test = obter_treino_teste(X, y)
    
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
    historico_acuracias = []

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
        # Guarda no histórico para o gráfico
        historico_acuracias.append(acc_priv)

        salvar_resultados(f"resultado_privado_eps_{eps}.txt",
                          f"Epsilon: {eps}\nRaio: {CONFIG['RAIO']}\nAcuracia: {acc_priv}\nPredicoes: {preds_priv.tolist()}")
    # 4. Gerar Gráfico
    gerar_grafico_comparativo(CONFIG["EPSILONS"], historico_acuracias, acc_trad)

    print("\n>>> Processo Finalizado.")