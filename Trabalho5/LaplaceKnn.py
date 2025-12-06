import numpy as np

"""
    Implementa o classificador r-N privado usando o Mecanismo de Laplace (Algoritmo 1).
    Usa o 'radius_r' para definir a vizinhança.
"""
class LaplaceKNN:

    def __init__(self, dataset, labels, privacy_budget, radius_r):
        self.dataset = dataset
        self.labels = labels
        self.privacy_budget = privacy_budget
        self.radius_r = radius_r

    def euclidiana(self, data_set, x_ponto):
        return np.linalg.norm(data_set - x_ponto, axis=1)

    def countXT(self, x_teste):
        """ Conta os rótulos dos vizinhos dentro do raio r para a amostra x_teste. """
        
        distancias = self.euclidiana(self.dataset, x_teste)

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

    def laplace(self, e):
        """
        Gera um ruído de Laplace manualmente usando Amostragem por Transformada Inversa.
        Fórmula: x = mu - b * sgn(u) * ln(1 - 2|u|)
        """
        # 1. Define a escala (b)
        b = 1 / e

        # 2. Gera um número aleatório uniforme (U) entre -0.5 e 0.5
        # np.random.rand() gera entre 0.0 e 1.0. Subtraindo 0.5, centralizamos no zero.
        u = np.random.rand() - 0.5

        # 3. Aplica a fórmula da Transformada Inversa
        # sgn pega o sinal (se u for negativo, o ruído vai para a esquerda, se positivo, direita)
        sinal = np.sign(u)

        # ln(1 - 2|u|) transforma a probabilidade linear em decaimento exponencial
        termo_log = np.log(1 - 2 * np.abs(u))
        ruido = -b * sinal * termo_log

        return ruido

    def calcular_contagens_ruidosas(self, x_teste, labelsSet):
        # Obter as contagens
        contagens = self.countXT(x_teste)

        # Se contagens estiver vazio (nenhum vizinho), para tudo e retorna None
        if not contagens:
            return None

        # Ajuste obrigatório: Composição Sequencial (Epsilon / L)
        L = len(labelsSet)
        epsilon_dividido = self.privacy_budget / L

        # Loop sobre todos os rótulos possíveis
        # Adiciona ruído de Laplace a cada contagem
        contagens_ruidosas = {}
        for I in labelsSet:
            contagem_I = contagens.get(I, 0)
            nc_I = contagem_I + self.laplace(epsilon_dividido)
            contagens_ruidosas[I] = nc_I

        return contagens_ruidosas


    def decidir_vencedor_ruidoso(self, contagens_ruidosas):
        # Retorna o rótulo com maior valor ruidoso
        label_previsto = max(contagens_ruidosas.items(), key=lambda item: item[1])[0]
        return label_previsto

    def atualizar_epsilon(self, epsilon):
        self.privacy_budget = epsilon