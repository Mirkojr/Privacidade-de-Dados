import pandas as pd
import json
import time 

# Definição e Geração das Hierarquias
hierarquia_idade = {
    "nível 0": {"faixa 1": [0, 0]},
    "nível 1": {"faixa 1": [0, 10], "faixa 2": [11, 20], "faixa 3": [21, 30],
                "faixa 4": [31, 40], "faixa 5": [41, 50], "faixa 6": [51, 60],
                "faixa 7": [61, 70], "faixa 8": [71, 80], "faixa 9": [81, 90],
                "faixa 10": [91, 100], "faixa 11": [101, 120]},
    "nível 2": {"faixa 1": [0, 20], "faixa 2": [21, 40], "faixa 3": [41, 60],
                "faixa 4": [61, 80], "faixa 5": [81, 100], "faixa 6": [101, 120]},
    "nível 3": {"faixa 1": [0, 40], "faixa 2": [41, 80], "faixa 3": [81, 120]},
    "nível 4": {"faixa 1": [0, 120]}
}
hierarquia_data = {
    "nível 0": "aaaa-mm-dd", "nível 1": "aaaa-mm", "nível 2": "aaaa"
}

with open("hierarquia_idade.json", "w", encoding="utf-8") as f:
    json.dump(hierarquia_idade, f, indent=4, ensure_ascii=False)
with open("hierarquia_data.json", "w", encoding="utf-8") as f:
    json.dump(hierarquia_data, f, indent=4, ensure_ascii=False)
print("Ficheiros hierarquia_idade.json e hierarquia_data.json gerados.")

# Funções de Generalização
def generalizar_idade(idade, nivel):
    if pd.isna(idade):
        return (0, 0)
    idade_int = int(idade) 
    if nivel == 0:
        return (idade_int, idade_int)
    for faixa, limites in hierarquia_idade[f"nível {nivel}"].items():
        if limites[0] <= idade_int <= limites[1]:
            return (limites[0], limites[1])
    return (0, 0)

# --- FUNÇÃO MODIFICADA PARA USAR pd.Timestamp ---
def generalizar_data(data, nivel):
    if pd.isna(data):
        return (0, 0)
    if nivel == 0:
        return (data, data)
    elif nivel == 1:
        # Usa pd.Timestamp em vez de datetime()
        start = pd.Timestamp(year=data.year, month=data.month, day=1)
        end = pd.Timestamp(year=data.year, month=data.month, day=28) 
        return (start, end)
    elif nivel == 2:
        # Usa pd.Timestamp em vez de datetime()
        start = pd.Timestamp(year=data.year, month=1, day=1)
        end = pd.Timestamp(year=data.year, month=12, day=31)
        return (start, end)
    else:
        return (0, 0)

# Funções de ajuda simples para calcular o tamanho da faixa
def tamanho_faixa_idade(faixa):
    return faixa[1] - faixa[0] + 1

def tamanho_faixa_data(faixa):
    if isinstance(faixa[0], int): # Lida com (0, 0)
        return 1
    # .days funciona tanto para timedelta do Python quanto para Timedelta do Pandas
    return (faixa[1] - faixa[0]).days + 1
    
# Leitura e Pré-processamento
try:
    df = pd.read_csv("dados_covid-ce_trab02.csv", low_memory=False)
    print(f"Dataset 'dados_covid-ce_trab02.csv' carregado com {len(df)} linhas.")
except FileNotFoundError:
    print("Erro: 'dados_covid-ce_trab02.csv' não encontrado. Encerrando.")
    exit()

print("Otimizando colunas de entrada (pré-processamento)...")
df['idadeCaso_limpa'] = pd.to_numeric(df['idadeCaso'], errors='coerce')
df['dataNascimento_limpa'] = pd.to_datetime(df['dataNascimento'], errors='coerce')
print("Pré-processamento concluído. Entrando no loop principal.")

# Loop Principal
while True:
    print("\n=== Anonimização de Dataset ===")
    try:
        ni = int(input("Digite o nível de generalização para idadeCaso (0–4, ou 9 para sair): "))
        nd = int(input("Digite o nível de generalização para dataNascimento (0–2, ou 9 para sair): "))
    except ValueError:
        print("Erro: Por favor, digite apenas números.")
        continue

    if ni == 9 and nd == 9:
        print("Encerrando o programa...")
        break
    
    if not (0 <= ni <= 4 or ni == 9) or not (0 <= nd <= 2 or nd == 9):
        print("Erro: Níveis inválidos. Por favor, tente novamente.")
        continue

    start_time = time.time()

    print("Processando generalização...")
    idade_faixa_col = df["idadeCaso_limpa"].apply(lambda x: generalizar_idade(x, ni))
    data_faixa_col = df["dataNascimento_limpa"].apply(lambda x: generalizar_data(x, nd))

    # CÁLCULO DE PRECISÃO
    print("Calculando precisão")
 
    hg_idade_col = idade_faixa_col.apply(tamanho_faixa_idade) / 121
    hg_data_col = data_faixa_col.apply(tamanho_faixa_data) / 365

    perda_total_media = ((hg_idade_col + hg_data_col) / 2).mean()
    precisao = 1 - perda_total_media


    # SAÍDA
    nome_dataset = f"DT_{ni}_{nd}.csv"
    print(f"Formatando colunas de saída para {nome_dataset}...")

    data_starts = data_faixa_col.str[0]

    mask_valid_dates = data_starts.apply(lambda x: isinstance(x, pd.Timestamp))
    valid_starts_dt = pd.to_datetime(data_starts[mask_valid_dates])
    
    data_formatada_col = pd.Series("(0, 0)", index=data_faixa_col.index, dtype=str)
    format_map = {0: "%Y-%m-%d", 1: "%Y-%m", 2: "%Y"}
    if nd in format_map:
        data_formatada_col.loc[mask_valid_dates] = valid_starts_dt.dt.strftime(format_map[nd])

    idade_formatada_col = idade_faixa_col.astype(str)
    if ni == 0:
        idade_formatada_col = idade_faixa_col.str[0]

    print(f"Gerando {nome_dataset}...")
    df_anon = pd.DataFrame({
        'idadeCaso': idade_formatada_col,
        'dataNascimento': data_formatada_col,
        'racaCor': df['racaCor'] 
    })
    
    df_anon.to_csv(nome_dataset, index=False)
    end_time = time.time()
    
    print(f"\n⏱️ Tempo de processamento: {end_time - start_time:.2f} segundos")
    print(f"✅ Dataset gerado: {nome_dataset}")
    print(f"→ Nível idadeCaso: {ni}, nível dataNascimento: {nd}")
    print(f"→ Precisão calculada: {precisao:.4f}")

print("\nPrograma finalizado.")