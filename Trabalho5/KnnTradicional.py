import numpy as np

def euclidiana(data_set, x_ponto):
        return np.linalg.norm(data_set - x_ponto, axis=1)

def knn_tradicional_k10(X_train, y_train, X_test, k=10):
        """
        Implementação simples do KNN clássico
        """
        predicoes = []
        total = len(X_test)
        
        print(f"Calculando KNN Tradicional (k={k})...")
        for i, x in enumerate(X_test):
            if i % 100 == 0: print(f"Progresso Tradicional: {i}/{total}")
            
            # 1. Distância
            dists = euclidiana(X_train, x)
            
            # 2. Pega os índices dos k menores
            # argsort ordena do menor pro maior
            k_indices = np.argsort(dists)[:k]
            
            # 3. Pega os rótulos
            k_labels = y_train[k_indices]
            
            # 4. Moda (Voto majoritário)
            # bincount conta quantas vezes cada número inteiro aparece
            contagem = np.bincount(k_labels)
            vencedor = np.argmax(contagem)
            predicoes.append(vencedor)
            
        return np.array(predicoes)