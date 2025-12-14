# Equipe

| Estudante | Matrícula |
|----------|----------|
| PEDRO JOAS FREITAS LIMA  | 548292  |
| MARCOS ANTONIO ALENCAR DA ROCHA JUNIOR  | 496563  |

# Descrição

## Tratamento 

No dataset dos candidatos, aplicamos algumas modificações, sendo elas:

**1. Remoção das aspas dentro do arquivo**
Acontece que algumas células tinham aspas inacabadas que afetavam a leitura, fazendo com que o Pandas juntasse uma coluna no lugar de duas. Isso trazia o problema seguinte.
**2. Linhas deslocadas**
Basicamente, um valor caia em outra coluna que não era a sua. Para isso, identificamos as posições que afetavam as variáveis que usaríamos e demos um shift para consertar
**3. Criação de um código único**
No dataset tínhamos informação de gênero, raça, município e data de nascimento. Para cada um, tinha um valor númerico associado, no caso do município, era usado o identificador da unidade eleitoral.

Já no dataset dos exames, aplicamos algumas modificações também, sendo elas:

**1. Padronização dos valores de raça**
**2. Substituição dos valores em texto pelo os valores númericos do dataset candidatos nas colunas gênero e raça**
**3. Linkar os municípios com seus identificadores da unidade eleitoral**
Nesse caso, usamos um [dataset público](https://raw.githubusercontent.com/betafcc/Municipios-Brasileiros-TSE/refs/heads/master/municipios_brasileiros_tse.csv) com esses identificadores.
**4. Também criando um código único**

## Ataque

Por fim, foi feito a junção dos datasets tratados, resultando o arquivo ```resultado_ataque.csv```. Foram encontrados 4757 correlações.

# Conclusão

Foi muito importante entendermos como é fácil reidentificar indivíduos e quanto alarmante isso é. Sendo assim, alcançamos o objetivo proposto no trabalho. Sobre a implementação, acreditamos que tinham outras formas de tratamento para melhor essa quantidade de pessoas, isso poderia ter sido resolvido com uma análise mais profunda de como os dados estavam organizados para encontrarmos um melhor tratamento. No mais, foi bem divertido!
