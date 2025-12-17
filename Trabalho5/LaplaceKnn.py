import numpy as np
from metrics import distancia_euclidiana

"""
    Implementa o classificador r-N privado usando o Mecanismo de Laplace (Algoritmo 1).
"""
class LaplaceKNN:

    def __init__(self, dataset, labels, privacy_budget, radius_r):
        """ Inicializa o classificador Laplace KNN. 
            Parâmetros:
            dataset: Conjunto de dados de treinamento (features).
            labels: Rótulos correspondentes ao conjunto de dados de treinamento.
            privacy_budget: Orçamento de privacidade (epsilon).
            radius_r: Raio r para considerar vizinhos.
        """
        self.dataset = dataset
        self.labels = labels
        self.privacy_budget = privacy_budget
        self.radius_r = radius_r


    def countXT(self, x_teste):
        """ Conta os rótulos dos vizinhos dentro do raio r para a amostra x_teste. """
        
        distancias = distancia_euclidiana(self.dataset, x_teste)

        # C_r_x(I) < r
        # np.where retorna uma tupla, onde o primeiro elemento é o array de índices
        indices_vizinhos = np.where(distancias <= self.radius_r)[0]

        # Se não houver vizinhos dentro do raio, retorna dicionário vazio
        if len(indices_vizinhos) == 0:
            return {}

        # Pegando os rótulos dos vizinhos dentro do raio
        rotulos_vizinhos = self.labels[indices_vizinhos]

        # Contagem dos votos dos rótulos dos vizinhos
        contagem_votos = {}
        for rotulo in rotulos_vizinhos:
            if rotulo in contagem_votos:
                contagem_votos[rotulo] += 1
            else:
                contagem_votos[rotulo] = 1

        return contagem_votos

    def gerar_ruido_laplace(self, epsilon: float) -> float:
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

    def calcular_contagens_ruidosas(self, x_teste, labelsSet):
        """ Calcula as contagens ruidosas para cada rótulo possível. """
        # Obter as contagens
        contagens = self.countXT(x_teste)

        # Se contagens estiver vazio (nenhum vizinho), para tudo, e retorna None
        if not contagens:
            return None

        # Ajuste obrigatório: Composição Sequencial (Epsilon / L)
        L = len(labelsSet)
        n = x_teste.shape[0]
        
        epsilon_dividido = self.privacy_budget / (L*n)

        # Loop sobre todos os rótulos possíveis
        # Adiciona ruído de Laplace a cada contagem
        contagens_ruidosas = {}
        for I in labelsSet:
            contagem_I = contagens.get(I, 0)
            nc_I = contagem_I + self.gerar_ruido_laplace(epsilon_dividido)
            contagens_ruidosas[I] = nc_I

        return contagens_ruidosas


    def decidir_vencedor_ruidoso(self, contagens_ruidosas):
        """ Decide o rótulo vencedor com base nas contagens ruidosas. """
        label_previsto = max(contagens_ruidosas.items(), key=lambda item: item[1])[0]
        return label_previsto

    def atualizar_epsilon(self, epsilon):
        """ Atualiza o orçamento de privacidade (epsilon). """
        self.privacy_budget = epsilon