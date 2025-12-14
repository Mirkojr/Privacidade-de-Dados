import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial, reduce

sns.set_style('darkgrid')

#realiza supressão dos dados de acordo com a porcentagem
def supressao(dados, porcentagem):
  df_copy = dados.copy()
  n_linhas = int(dados.shape[0] * porcentagem)
  indexes = df_copy.sample(n_linhas).index
  df_copy.loc[indexes, 'municipioCaso'] = 0

  return df_copy

#função para plotar os 20 melhores
def top_20_plot(dados, porcentagem):
  vinte_melhores = dados['municipioCaso'].value_counts().reset_index().query('municipioCaso != 0').iloc[:20]

  #plt.figure(figsize=(12,8))
  ax = sns.barplot(vinte_melhores, y='municipioCaso', x='count')
  ax.bar_label(ax.containers[0])

  plt.title(f'Frequência de municípios ({porcentagem}%)')
  plt.xlabel('Frequência')
  plt.ylabel('Município')
  plt.grid(True, alpha=0.5)
  
df_0 = pd.read_csv('/content/dados_covid-ce_trab02.csv', low_memory=False)

df_25 = supressao(df_0, 0.25)
df_50 = supressao(df_0, 0.50)
df_75 = supressao(df_0, 0.75)

dfs = [df_0, df_25, df_50, df_75]
porcentagens = [0, 25, 50, 75]

#realizar contagens 
contagem = df_0['municipioCaso'].value_counts().reset_index().query('municipioCaso != 0').sort_values('municipioCaso').rename(columns={'count':'Frequencia_original'})

contagem_25 = df_25['municipioCaso'].value_counts().reset_index().query('municipioCaso != 0').sort_values('municipioCaso').rename(columns={'count':'Frequencia_25'})
contagem_50 = df_50['municipioCaso'].value_counts().reset_index().query('municipioCaso != 0').sort_values('municipioCaso').rename(columns={'count':'Frequencia_50'})
contagem_75 = df_75['municipioCaso'].value_counts().reset_index().query('municipioCaso != 0').sort_values('municipioCaso').rename(columns={'count':'Frequencia_75'})

dfs_contagens = [contagem, contagem_25, contagem_50, contagem_75]
merge = partial(pd.merge, on=['municipioCaso'], how='outer')
contagens_merge_reduce = reduce(merge, dfs_contagens)
contagens_merge_reduce.to_csv('frequencias trab2.csv', index=False)

#Salvar arquivos 
for df, porcentagem in zip(dfs[1:], porcentagens[1:]):
  df.to_csv(f'dados_covid-ce_trab02_{porcentagem}.csv', index=False)
  
#plotar gráficos
fig, axs = plt.subplots(2, 2, figsize=(24, 12))
axs = axs.flatten()

for i, (df, porcentagem) in enumerate(zip(dfs, porcentagens)):
    plt.sca(axs[i])
    top_20_plot(df, porcentagem)

