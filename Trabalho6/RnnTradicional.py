import numpy as np
from metrics import distancia_euclidiana

def rnn_tradicional(X_train, y_train, X_test, radius):
    """
    Implementação do Radius-Neighbors (r-NN) Tradicional.
    Classifica baseado na moda das classes dentro do raio r.
    """
    predicoes = []
    
    classes = np.unique(y_train)
    
    for x in X_test:
        # Distância
        dists = distancia_euclidiana(X_train, x)

        # Vizinhos no raio r
        indices_no_raio = np.where(dists <= radius)[0]
        
        # Se não tiver vizinhos, escolhemos um aleatório
        if len(indices_no_raio) == 0:
            predicoes.append(np.random.choice(classes))
            continue
            
        # Pega os rótulos
        labels_no_raio = y_train[indices_no_raio]
        
        # Moda (Voto majoritário)
        contagem = np.bincount(labels_no_raio)
        vencedor = np.argmax(contagem)
        predicoes.append(vencedor)
            
    return np.array(predicoes)