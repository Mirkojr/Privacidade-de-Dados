import numpy as np
from metrics import distancia_euclidiana

class ExponentialKNN:
    def __init__(self, dataset, labels, privacy_budget_total, radius_r):
        self.dataset = dataset
        self.labels = labels
        self.epsilon_total = privacy_budget_total
        self.radius_r = radius_r

    def atualizar_params(self, epsilon_total, radius_r):
        self.epsilon_total = epsilon_total
        self.radius_r = radius_r

    def predict(self, X_test, classes_unicas):
        n_queries = len(X_test)
        
        # Composição Sequencial: Dividimos o orçamento pelo número de consultas (tamanho do teste)
        # Sensibilidade Global (Delta u) = 1 (adicionar/remover um registro altera a contagem em no máx 1)
        # Fórmula Exponencial: epsilon_query = epsilon_total / N
        epsilon_query = self.epsilon_total / n_queries
        sensitivity = 1.0
        
        predicoes = []

        for x in X_test:
            # - Calcular distâncias e encontrar vizinhos no raio
            dists = distancia_euclidiana(self.dataset, x)
            indices_vizinhos = np.where(dists <= self.radius_r)[0]
            
            # Se não tem vizinhos, retornamos um aleatorio
            if len(indices_vizinhos) == 0:
                predicoes.append(np.random.choice(classes_unicas))
                continue

            rotulos_vizinhos = self.labels[indices_vizinhos]

            # - Função de Utilidade u(x, c): Contagem de cada classe
            # Inicializa scores zerados para todas as classes
            scores = {c: 0 for c in classes_unicas}
            for rotulo in rotulos_vizinhos:
                scores[rotulo] += 1
            
            # Converter scores para array alinhado com classes_unicas
            score_array = np.array([scores[c] for c in classes_unicas])

            # - Calcular Probabilidades do Mecanismo Exponencial
            # P(c) proportional to exp( (epsilon * u(c)) / (2 * delta_u) )
            
            # Subtraindo o max score para evitar overflow
            # Isso não altera a probabilidade final pois cancela no numerador/denominador
            exponent = (epsilon_query * score_array) / (2 * sensitivity)
            exponent = exponent - np.max(exponent) 
            
            probs_nao_normalizadas = np.exp(exponent)
            probs_finais = probs_nao_normalizadas / np.sum(probs_nao_normalizadas)

            # - Amostrar a classe baseada nas probabilidades
            classe_escolhida = np.random.choice(classes_unicas, p=probs_finais)
            predicoes.append(classe_escolhida)

        return np.array(predicoes)