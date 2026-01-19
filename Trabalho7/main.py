import os
import numpy as np
import pandas as pd
import kagglehub
from typing import Tuple, Dict
import matplotlib.pyplot as plt 

from metodos import resposta_randomizada, estimativa_moedas_justas

# --- CONFIGURAÇÃO ---
CONFIG = {
    "REPO_URL": "wenruliu/adult-income-dataset",
    "FILENAME": "adult.csv",
    "TARGET_COL": "income",
    "GRAPH_FILE": "grafico_acuracia_exponencial.png",
    # Nomes dos arquivos
    "FILE_TRAD": "RELATORIO_TRADICIONAL.txt",
    "FILE_PRIV": "RELATORIO_EXPONENCIAL.txt"
}

def carregar_dataset() -> pd.Series:
    """Baixa e carrega o dataset."""
    try:
        path = kagglehub.dataset_download(CONFIG["REPO_URL"])
        file_path = os.path.join(path, CONFIG["FILENAME"])
        return pd.read_csv(file_path, usecols=['income'])['income']
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar dataset: {e}")

# --- MAIN ---
if __name__ == "__main__":

    respostas = carregar_dataset().to_list()
    respostas_randomizadas = []
    for resposta in respostas:
        respostas_randomizadas.append(resposta_randomizada(resposta))
    
    estimativa = estimativa_moedas_justas(respostas_randomizadas)
    print(estimativa)

    # # Loop Principal
    # for r in CONFIG["RADIUS"]:
    #     print(f"\n==== Processando Raio r = {r} ====")
        
    #     # A) Rodar Tradicional
    #     preds_trad = rnn_tradicional(X_train, y_train, X_test, radius=r)
    #     acc_trad = np.mean(preds_trad == y_test)
    #     hist_acc_tradicional[r] = acc_trad
    #     print(f"   [Tradicional] Acurácia: {acc_trad:.4f}")
        
    #     # Acumula no buffer tradicional
    #     buffer_tradicional += f"--- RAIO: {r} ---\n"
    #     buffer_tradicional += f"Acuracia: {acc_trad:.4f}\n"
    #     buffer_tradicional += f"Predicoes (primeiras 50): {preds_trad[:50].tolist()} ... [truncado]\n"
    #     buffer_tradicional += "="*40 + "\n\n"

    #     # B) Rodar Privado
    #     for eps in CONFIG["EPSILONS"]:
    #         knn_exp.atualizar_params(epsilon_total=eps, radius_r=r)
    #         preds_priv = knn_exp.predict(X_test, classes_unicas)
    #         acc_priv = np.mean(preds_priv == y_test)
    #         hist_acc_privado[r].append(acc_priv)
            
    #         print(f"   [Exponencial] eps={eps}: Acurácia={acc_priv:.4f}")
            
    #         # Acumula no buffer privado
    #         buffer_privado += f"--- RAIO: {r} | EPSILON: {eps} ---\n"
    #         buffer_privado += f"Acuracia: {acc_priv:.4f}\n"
    #         buffer_privado += f"Predicoes (primeiras 50): {preds_priv[:50].tolist()} ... [truncado]\n"
    #         buffer_privado += "-"*30 + "\n\n"

    # # - Salvar os arquivos consolidados
    # print("\n>>> Salvando relatórios...")
    # salvar_resultados(CONFIG["FILE_TRAD"], buffer_tradicional)
    # salvar_resultados(CONFIG["FILE_PRIV"], buffer_privado)
    # print(f"   -> {CONFIG['FILE_TRAD']} salvo com sucesso.")
    # print(f"   -> {CONFIG['FILE_PRIV']} salvo com sucesso.")

    # - Gerar Gráfico
    # gerar_grafico_comparativo(hist_acc_privado, hist_acc_tradicional)
    # print("\n>>> Fim da Execução.")