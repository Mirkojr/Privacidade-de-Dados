import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

# --- SUA FUNÇÃO DE RUÍDO (Simulada aqui fora da classe para teste) ---

# Para o teste, vamos definir a função fora da classe, apenas como uma função pura.
def gerar_ruido_laplace_pura(epsilon: float) -> float:
        """
        Gera um ruído de Laplace usando Amostragem por Transformada Inversa.
        Fórmula: x = mu - b * sgn(u) * ln(1 - 2|u|)
        """
        escala = 1 / epsilon

        u = np.random.rand() - 0.5
        sinal = np.sign(u)
        termo_log = np.log(1 - 2 * np.abs(u))

        ruido = -escala * sinal * termo_log
        return ruido


# --- FUNÇÃO DE TESTE ---

def testar_distribuicao_laplace(epsilon: float, n_amostras: int = 100000, filename: str = "distribuicao_laplace.png"):
    """
    Gera amostras de ruído de Laplace e compara com a PDF teórica.
    
    Parâmetros:
    epsilon: O orçamento de privacidade (epsilon).
    n_amostras: Número de amostras de ruído para gerar.
    filename: Nome do arquivo para salvar o gráfico.
    """
    print(f"\n--- Testando Distribuição de Laplace para epsilon = {epsilon} ---")
    
    # 1. Geração das amostras
    # Chamamos a função muitas vezes
    amostras_ruido = [gerar_ruido_laplace_pura(epsilon) for _ in range(n_amostras)]
    
    # 2. Configurações Teóricas de Laplace
    # O parâmetro 'b' na função laplace de scipy é a 'escala' (b = Delta f / epsilon)
    escala_teorica = 1.0 / epsilon 
    
    # 3. Plotagem
    plt.figure(figsize=(12, 6))
    
    # Histograma dos dados gerados (Sua implementação)
    # bins='auto' deixa o matplotlib escolher o melhor número de barras
    plt.hist(amostras_ruido, bins=100, density=True, alpha=0.6, color='skyblue', label='Amostras Geradas (Sua função)')
    
    # Curva Teórica (PDF de Laplace)
    # Cria pontos para plotar a curva suave
    x = np.linspace(min(amostras_ruido), max(amostras_ruido), 1000)
    pdf_teorica = laplace.pdf(x, loc=0, scale=escala_teorica)
    
    plt.plot(x, pdf_teorica, 'r-', lw=2, label=f'PDF Teórica Laplace (b = 1/{epsilon:.2f})')
    


    
    # Configurações do Gráfico
    plt.title(f'Distribuição do Ruído de Laplace (ε={epsilon}, b={escala_teorica:.3f})')
    plt.xlabel('Ruído Adicionado')
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Salvar e Mostrar
    plt.savefig(filename)
    print(f"Resultado do teste salvo em: {filename}")
    plt.close()

# --- EXECUÇÃO DO TESTE ---
if __name__ == "__main__":
    # Testar com um epsilon 'fácil' de visualizar
    testar_distribuicao_laplace(epsilon=1.0, filename="laplace_e1_0.png")
    
    # Testar com um epsilon 'apertado' (mais ruído, curva mais espalhada)
    testar_distribuicao_laplace(epsilon=0.1, filename="laplace_e0_1.png")
    
    # Testar com um epsilon 'relaxado' (menos ruído, curva mais alta e pontiaguda)
    testar_distribuicao_laplace(epsilon=5.0, filename="laplace_e5_0.png")
    
    print("\nTestes concluídos. Verifique os arquivos PNG gerados.")