from numpy.random import choice
def resposta_randomizada(resposta, p=0.5, q=0.5):
    moedas = ['Cara', 'Coroa']
    moeda_escolhida = choice(moedas, p=[p,q])
    resposta_verdadeira = 'Sim' if resposta == '>50K' else 'Não'
    
    if moeda_escolhida == 'Cara':
        return resposta_verdadeira
    else:
        moeda_escolhida = choice(moedas)
        if moeda_escolhida == 'Cara':
            return 'Sim'
        else:
            return 'Não'

def estimativa_moedas_justas(vetor_respostas):
    S = vetor_respostas.count('Sim')
    n = len(vetor_respostas)
    T = 2*S - int(n/2)

    return T if T >= 0 else 0

if __name__ == '__main__':
    respostas = ['>50K', '<=50K', '>50K', '<=50K', '<=50K']

    respostas_randomizadas = []
    for resposta in respostas:
        respostas_randomizadas.append(resposta_randomizada(resposta))

    estimativa = estimativa_moedas_justas(respostas_randomizadas)
    print(estimativa)
