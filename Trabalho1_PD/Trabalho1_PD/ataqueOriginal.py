import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from unicodedata import normalize

# TRATAMENTO DOS DADOS DOS CANDIDATOS
def conserta_arquivo(caminho, novo_caminho):
    with open(caminho, encoding='latin-1') as f:
      arquivo = f.read()

    arquivo_formatado = arquivo.replace('"', '')

    with open(novo_caminho, 'w', encoding='latin-1') as f:
      f.write(arquivo_formatado)

def tratamento_candidatos(candidatos):

  # essas linhas em específico estavam com valores empurrados para o lado, para consertar demos um shift
  candidatos.loc[candidatos['CD_GENERO'] == 'DEFERIDO', :] = candidatos.loc[candidatos['CD_GENERO'] == 'DEFERIDO', :].astype(str).shift(18, axis=1)

  candidatos.loc[candidatos['NM_COLIGACAO'].str.contains('»', na=False), candidatos.columns[30:]] = (
      candidatos.loc[candidatos['NM_COLIGACAO'].str.contains('»', na=False), candidatos.columns[30:]].shift(axis=1)
  )

  candidatos_selecionados = candidatos[['NM_CANDIDATO','DT_NASCIMENTO', 'CD_COR_RACA', 'CD_GENERO', 'SG_UE']].copy()

  # aqui é feito a junção dos valores, queríamos fazer apenas usando os números, pois os munícipios estavam com problema de caractere
  candidatos_selecionados['codigo_referencia'] = (candidatos_selecionados['DT_NASCIMENTO'].str.replace('/', '') + candidatos_selecionados['CD_COR_RACA'] +
                                                candidatos_selecionados['CD_GENERO'].astype(str) + candidatos_selecionados['SG_UE'].astype(str))

  candidatos_selecionados.dropna(inplace=True) # 16213 -> 16194, 19 registros removidos pois tinham valores nulos
  return candidatos_selecionados

def pega_dados_unidade_eleitoral():
  # isso aqui é para converter os nomes dos municipios para os codigos da unidade eleitoral do tse que bate com o outro df

  codigos_ue = pd.read_csv('https://raw.githubusercontent.com/betafcc/Municipios-Brasileiros-TSE/refs/heads/master/municipios_brasileiros_tse.csv')
  codigos_ue = codigos_ue[codigos_ue['uf'] == 'CE'][['codigo_tse', 'nome_municipio']]
  return codigos_ue

#TRATAMENTO DOS DADOS DOS EXAMES
def ataque(candidatos, candidatos_selecionados):

  exames = pd.read_csv('dados_covid-ce_trab02.csv', low_memory=False)
  exames_selecionados = exames[['identificadorCaso', 'dataNascimento', 'racaCor', 'sexoCaso', 'municipioCaso', 'dataInicioSintomas', 'resultadoFinalExame']].copy()
  exames_selecionados['racaCor'] = exames_selecionados['racaCor'].str.upper()
  # aqui foi feito a correlação dos códigos usados no outro dataset para ficarem iguais, tanto para a coluna gênero quanto a coluna raça
  racas_codigos = candidatos[['DS_COR_RACA', 'CD_COR_RACA']].dropna().drop_duplicates().replace('SEM INFORMAÃO', 'SEM INFORMACAO')
  genero_codigos = candidatos[['DS_GENERO', 'CD_GENERO']].dropna().drop_duplicates()

  exames_cor = pd.merge(exames_selecionados, racas_codigos, left_on='racaCor', right_on='DS_COR_RACA').drop(['DS_COR_RACA', 'racaCor'], axis=1)
  exames_genero = pd.merge(exames_cor, genero_codigos, left_on='sexoCaso', right_on='DS_GENERO').drop(['DS_GENERO', 'sexoCaso'], axis=1)

  
  codigos_ue = pega_dados_unidade_eleitoral()

  codigos_ue['nome_municipio'] = codigos_ue['nome_municipio'].apply(lambda x: normalize('NFKD', x).encode('ASCII','ignore').decode('ASCII')) # remove acentos isso aqui viu
  exames_tse = pd.merge(exames_genero, codigos_ue, left_on='municipioCaso', right_on='nome_municipio').drop(['nome_municipio', 'municipioCaso'], axis=1)
  exames_tse['dataNascimento'] = pd.to_datetime(exames_tse['dataNascimento'], format='%Y-%m-%d', errors='coerce')
  exames_tse['dataNascimento'] = exames_tse['dataNascimento'].dt.strftime('%d/%m/%Y')
  exames_tse['codigo_referencia'] = (exames_tse['dataNascimento'].str.replace('/', '') + exames_tse['CD_COR_RACA'] +
                                                exames_tse['CD_GENERO'] + exames_tse['codigo_tse'].astype(str))
  exames_candidatos = pd.merge(candidatos_selecionados, exames_tse[['codigo_referencia', 'dataInicioSintomas', 'resultadoFinalExame']], on='codigo_referencia')
  print('Quantidade de correlações encontradas: ', exames_candidatos.shape[0])

  exames_candidatos.to_csv('resultado_ataque.csv', index=False)


if __name__ == '__main__':
  cand_caminho = 'consulta_cand_2020_CE.csv'
  cand_caminho_novo = 'consulta_cand_2020_CE_limpo.csv'
  conserta_arquivo(caminho=cand_caminho, novo_caminho=cand_caminho_novo)

  candidatos = pd.read_csv(cand_caminho_novo, sep=';', encoding='latin-1')

  candidatos_selecionados = tratamento_candidatos(candidatos)

  ataque(candidatos, candidatos_selecionados)
