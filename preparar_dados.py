import pandas as pd

print("⏳ Lendo e inspecionando o arquivo original...")
# Carrega o arquivo usando o separador ';' que identificamos
df = pd.read_csv("dados_historicos.csv", sep=";")

# 1. Limpar espaços extras nos nomes de todas as colunas
df.columns = df.columns.str.strip()

# 2. Corrigir a coluna de valores (remover pontos, trocar vírgulas por pontos)
print("🧹 Limpando os valores financeiros...")
df['Valor Gasto'] = df['Valor Gasto'].astype(str).str.replace('.', '', regex=False)
df['Valor Gasto'] = df['Valor Gasto'].astype(str).str.replace(',', '.', regex=False)
df['Valor Gasto'] = pd.to_numeric(df['Valor Gasto'], errors='coerce')

# 3. Corrigir a coluna de datas
print("📅 Formatando as datas...")
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')

# Remover linhas que possam ter ficado em branco/com erro após a conversão
df = df.dropna(subset=['Data', 'Valor Gasto'])

# 4. Agrupar os gastos por DIA (Granularidade diária)
print("📊 Agrupando gastos por dia...")
df_diario = df.groupby('Data')['Valor Gasto'].sum().reset_index()

# 5. Renomear para o padrão do Prophet
df_diario.columns = ['ds', 'y']

# Salvar o resultado diário
df_diario.to_csv("gastos_limpos.csv", index=False)

# Mostrar uma prévia de como ficou
print("\nPrévia dos seus gastos mensais acumulados:")
print(df_diario.tail(10))