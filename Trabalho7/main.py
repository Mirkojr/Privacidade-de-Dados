import os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt 

from metodos import *

# --- CONFIGURAÇÃO ---
CONFIG = {
    "REPO_URL": "wenruliu/adult-income-dataset",
    "FILENAME": "adult.csv",
    "GRAPH_FILE": "grafico_acuracia_exponencial.png",
    "p": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    "q": [0.9, 0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0],
}

def carregar_dataset() -> pd.Series:
    """Baixa e carrega o dataset."""
    try:
        path = kagglehub.dataset_download(CONFIG["REPO_URL"])
        file_path = os.path.join(path, CONFIG["FILENAME"])
        return pd.read_csv(file_path, usecols=['income'])['income']
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar dataset: {e}")

def consulta(resposta):
    if resposta == '>50K':
        return 'Sim'
    else:
        return 'Não'
    
def experimento(respostas_verdadeiras, p, q, k=10):
    estimativa_real = np.count_nonzero(respostas_verdadeiras == 'Sim')
    erros = []
    for _ in range(k):
        respostas_randomizadas = []
        for resposta in respostas_verdadeiras:
            respostas_randomizadas.append(resposta_randomizada(resposta, p, q))

        acuracia = np.mean((respostas_verdadeiras == respostas_randomizadas).astype(int))
        estimativa = estimador(np.array(respostas_randomizadas), p=p, q=q)

        erros.append(abs(estimativa_real - estimativa))

    return np.mean(erros), np.mean(acuracia)

def plota_grafico(valores_p, metrica, ylabel, title, fname):
    plt.figure()
    plt.plot(valores_p, metrica)

    plt.xlabel('Valores das probabilidades (p)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname, dpi=600)
    plt.close()

# --- MAIN ---
if __name__ == "__main__":

    print('------------ Carregando dataset ------------')
    respostas_verdadeiras = carregar_dataset().to_numpy()
    respostas_verdadeiras = np.array([consulta(resposta) for resposta in respostas_verdadeiras])

    print('------------ Calculando experimento para moedas justas ------------')
    
    mae_justas, acuracia_justas = experimento(respostas_verdadeiras, p=0.5, q=0.5)
    print('Métricas para o caso das moedas justas.')
    print(f'MAE: {mae_justas}, Acuracia: {acuracia_justas}')

    print('------------ Calculando experimento para moedas tendenciosas ------------')
    p=0.6
    q=0.4

    mae_tendenciosas, acuracia_tendenciosas = experimento(respostas_verdadeiras, p=p, q=q)
    print(f'Métricas para o caso das moedas tendenciosas com p={p} e q={q}.')
    print(f'MAE: {mae_tendenciosas}, Acuracia: {acuracia_tendenciosas}')

    print('------------ Calculando experimento para diferentes valores de p e q ------------')
    maes = []
    acuracias = []
    for valor_p, valor_q in list(zip(CONFIG['p'], CONFIG['q'])):
        print(f'p={valor_p}, q={valor_q}')
        mae, acuracia = experimento(respostas_verdadeiras, p=valor_p, q=valor_q)
        print(f'MAE: {mae}, Acuracia: {acuracia}')
        maes.append(mae)
        acuracias.append(acuracia)
        print('--'*10)

    plota_grafico(valores_p=CONFIG['p'], metrica=maes, 
                  ylabel='MAE (Erro absoluto médio)', title='Comportamento do erro para diferentes probabilidades', 
                  fname='plot_mae_valores_p.png')
    plota_grafico(valores_p=CONFIG['p'], metrica=acuracias, 
                  ylabel='Acurácia', title='Comportamento da acurácia para diferentes probabilidades',
                  fname='plot_acc_valores_p.png')

   