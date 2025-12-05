import sys
import os
# Ajuste de path para importar a classe
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from LaplaceKnn import LaplaceKNN
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter 

def gerar_relatorio_completo(dataset, labels, x_teste, labels_set, radius_r, privacy_budget, nome_teste):
    """
    Gera um painel com 3 gráficos:
    1. Snapshot de uma execução única (Barras).
    2. Histograma das distribuições (100 execuções).
    3. Contagem de vitórias (100 execuções).
    """
    n_execucoes = 100
    
    laplace_knn = LaplaceKNN(dataset=dataset, 
                             labels=labels, 
                             privacy_budget=privacy_budget, 
                             radius_r=radius_r)

    historico_votos = {label: [] for label in labels_set}
    historico_previsoes = [] 
    ultimo_resultado_snapshot = None # Vai guardar a última rodada para plotar o snapshot
    
    print(f"\nProcessando {nome_teste} ({n_execucoes} simulações)...")
    
    for i in range(n_execucoes):
        resultado = laplace_knn.calcular_contagens_ruidosas(x_teste, labels_set)
        
        # Guarda o resultado da última rodada para o gráfico de Snapshot
        if i == n_execucoes - 1:
            ultimo_resultado_snapshot = resultado

        if resultado:
            for label, valor_ruidoso in resultado.items():
                historico_votos[label].append(valor_ruidoso)
            
            vencedor = laplace_knn.decidir_vencedor_ruidoso(resultado)
            historico_previsoes.append(vencedor)
        else:
            historico_previsoes.append("Nenhum")

    # --- CONFIGURAÇÃO DA PLOTAGEM (3 Subplots) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Cores padronizadas
    cores_base = {'A': 'blue', 'B': 'red', 'C': 'green', 'X': 'blue', 'Y': 'red', 'Z': 'gray', 
                  'Nenhum': 'gray', 'Cubo': 'purple', 'Esfera': 'orange', 'Origem': 'cyan'}

    # === GRÁFICO 1: SNAPSHOT (Última Rodada) ===
    if ultimo_resultado_snapshot:
        classes_snap = list(ultimo_resultado_snapshot.keys())
        votos_snap = list(ultimo_resultado_snapshot.values())
        vencedor_snap = laplace_knn.decidir_vencedor_ruidoso(ultimo_resultado_snapshot)
        
        # Destaca o vencedor com cor cheia, outros com transparência
        cores_snap = [cores_base.get(cls, 'gray') if cls == vencedor_snap else 'lightgray' for cls in classes_snap]
        edge_colors = [cores_base.get(cls, 'black') for cls in classes_snap]

        barras1 = ax1.bar(classes_snap, votos_snap, color=cores_snap, edgecolor=edge_colors, linewidth=2)
        
        ax1.set_title(f'Snapshot (1 Rodada Exemplo)\nVencedor: {vencedor_snap}')
        ax1.set_ylabel('Votos (Real + Ruído)')
        ax1.axhline(0, color='black', linewidth=0.8)
        
        # Valores nas barras
        for barra in barras1:
            h = barra.get_height()
            pos_y = h + (0.05 if h >= 0 else -0.15)
            ax1.text(barra.get_x() + barra.get_width()/2., pos_y, f'{h:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Sem Vizinhos', ha='center', va='center')

    # === GRÁFICO 2: HISTOGRAMA (Distribuição) ===
    
    tem_dados = False
    for label in labels_set:
        dados = historico_votos[label]
        if dados:
            tem_dados = True
            ax2.hist(dados, bins=15, alpha=0.5, label=f'{label}', 
                     color=cores_base.get(label, 'gray'), edgecolor='black')
    
    if tem_dados: ax2.legend()
    ax2.set_title(f'Distribuição dos Votos (100 Execuções)\nEPS: {privacy_budget} | Raio: {radius_r}')
    ax2.set_xlabel('Valor do Voto')

    # === GRÁFICO 3: VITÓRIAS (Resultado Final) ===
    contagem_final = Counter(historico_previsoes)
    classes_win = list(contagem_final.keys())
    qtds_win = list(contagem_final.values())
    cores_win = [cores_base.get(cls, 'gray') for cls in classes_win]

    barras3 = ax3.bar(classes_win, qtds_win, color=cores_win, edgecolor='black')
    
    ax3.set_title(f'Total de Vitórias')
    ax3.set_ylabel('Qtd Vitórias')

    for barra in barras3:
        h = barra.get_height()
        ax3.text(barra.get_x() + barra.get_width()/2., h, f'{int(h)}', ha='center', va='bottom')

    # Finalização
    plt.tight_layout()
    
    nome_arquivo = f'painel_{nome_teste}.png'
    if os.path.exists('testes'):
        nome_arquivo = f'testes/painel_{nome_teste}.png'
        
    plt.savefig(nome_arquivo)
    print(f"Painel salvo como: {nome_arquivo}")
    plt.show()

# --- DEFINIÇÃO DOS TESTES (Só dados e chamada) ---

def Teste1():
    print("\n=== Teste 1: Cenário Básico ===")
    D_data = np.array([[1.0, 1.0], [1.1, 1.2], [5.0, 5.0], [5.2, 5.1], [2.0, 2.0]])
    L_labels = np.array(['A', 'A', 'B', 'B', 'C'])
    x_teste = np.array([1.5, 1.5])
    labels_set = set(['A', 'B', 'C'])
    
    gerar_relatorio_completo(D_data, L_labels, x_teste, labels_set, radius_r=1.0, privacy_budget=0.5, nome_teste="Teste1")

def Teste2():
    print("\n=== Teste 2: Empate Técnico ===")
    D_data = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [5.0, 5.0]])
    L_labels = np.array(['X', 'X', 'Y', 'Y', 'Z'])
    x_teste = np.array([0.0, 0.0])
    labels_set = set(['X', 'Y', 'Z'])
    
    gerar_relatorio_completo(D_data, L_labels, x_teste, labels_set, radius_r=1.1, privacy_budget=0.5, nome_teste="Teste2_Empate")

def Teste3():
    print("\n=== Teste 3: Sem Vizinhos no Raio ===")
    D_data = np.array([[10.0, 10.0], [20.0, 20.0], [-10.0, -5.0]])
    L_labels = np.array(['A', 'A', 'B'])
    x_teste = np.array([0.0, 0.0])
    labels_set = set(['A', 'B'])
    
    gerar_relatorio_completo(D_data, L_labels, x_teste, labels_set, radius_r=5.0, privacy_budget=1.0, nome_teste="Teste3_Vazio")

def Teste4():
    print("\n=== Teste 4: Dados em 3D ===")
    D_data = np.array([
        [1.0, 1.0, 1.0], [1.1, 1.1, 0.9], 
        [5.0, 5.0, 5.0], [0.0, 0.0, 0.0], 
        [0.5, 0.5, 0.5], [10.0, 10.0, 10.0]
    ])
    L_labels = np.array(['Cubo', 'Cubo', 'Esfera', 'Origem', 'Origem', 'Esfera'])
    x_teste = np.array([10.0, 10.0, 10.0])
    labels_set = set(['Cubo', 'Esfera', 'Origem'])
    
    gerar_relatorio_completo(D_data, L_labels, x_teste, labels_set, radius_r=0.5, privacy_budget=2.0, nome_teste="Teste4_3D")

def Teste5():
    print("\n=== Teste 5: Repetição do Básico ===")
    # Mesmos dados do Teste 1
    D_data = np.array([[1.0, 1.0], [1.1, 1.2], [5.0, 5.0], [5.2, 5.1], [2.0, 2.0]])
    L_labels = np.array(['A', 'A', 'B', 'B', 'C'])
    x_teste = np.array([1.5, 1.5])
    labels_set = set(['A', 'B', 'C'])
    
    gerar_relatorio_completo(D_data, L_labels, x_teste, labels_set, radius_r=1.0, privacy_budget=0.5, nome_teste="Teste5_Completo")

def laplaceTeste():
    print("\n=== Teste de Geração de Ruído ===")
    e = 0.5
    laplace_knn = LaplaceKNN(dataset=None, labels=None, privacy_budget=e, radius_r=None)
    vals = []
    for i in range(1000):
        vals.append(laplace_knn.laplace(e))
    
    plt.figure(figsize=(8,5))
    plt.hist(vals, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.title("Distribuição do Ruído Gerado (Teste)")
    plt.show()

if __name__ == "__main__":
    n_teste = input("Digite o número do teste (1-5) ou 'laplace': ")

    match n_teste:
        case '1': Teste1()
        case '2': Teste2()
        case '3': Teste3()
        case '4': Teste4()
        case '5': Teste5()
        case 'laplace': laplaceTeste()
        case _: print("Teste inválido.")