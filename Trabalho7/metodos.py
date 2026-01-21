import numpy as np

def resposta_randomizada(resposta, p_cara=0.5, q_coroa=0.5):
    moedas = ['Cara', 'Coroa']
    moeda_escolhida = np.random.choice(moedas, p=[p_cara,q_coroa])
    
    if moeda_escolhida == 'Cara':
        return resposta
    else:
        moeda_escolhida = np.random.choice(moedas, p=[p_cara,q_coroa])
        if moeda_escolhida == 'Cara':
            return 'Sim'
        else:
            return 'Não'

def estimador(vetor_respostas_randomizadas, p=0.5, q=0.5):
    S = np.count_nonzero(vetor_respostas_randomizadas == 'Sim')
    n = len(vetor_respostas_randomizadas)
    T = (S - n * q*p)/p

    return round(T) if T >= 0 else 0

if __name__ == '__main__':
    
    respostas = np.array(['>50K', '<=50K', '>50K', '<=50K', '<=50K'])
    func_vec = np.vectorize(resposta_randomizada)
    respostas_randomizadas = func_vec(respostas)
    # for resposta in respostas:
    #     respostas_randomizadas.append(resposta_randomizada(resposta))

    print(respostas)
    print(respostas_randomizadas)
    estimativa = estimador(respostas_randomizadas)
    print(np.count_nonzero(respostas == '>50K'))
    print(estimativa)
