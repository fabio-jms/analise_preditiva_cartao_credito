import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURAÇÃO DE GRANULARIDADE
# Mude para 'MENSAL' ou 'DIARIO' dependendo do que quer analisar agora
GRANULARIDADE = 'MENSAL' 
# =====================================================================

print(f"⏳ Carregando os dados no modo: {GRANULARIDADE}...")

if GRANULARIDADE == 'MENSAL':
    df = pd.read_csv("gastos_limpos_mensal.csv")
    # Configurações ideais para dados mensais
    periods_to_predict = 12  # Prever os próximos 12 meses
    frequency = 'MS'         # Frequência Mensal (Início do Mês)
    weekly_season = False    # Não faz sentido olhar dia da semana em dados mensais
elif GRANULARIDADE == 'DIARIO':
    df = pd.read_csv("gastos_limpos_diario.csv")
    # Configurações ideais para dados diários
    periods_to_predict = 90  # Prever os próximos 90 dias
    frequency = 'D'          # Frequência Diária
    weekly_season = True     # Ativa o efeito "fim de semana"
else:
    print("❌ Modo inválido! Escolha 'MENSAL' ou 'DIARIO'.")
    exit()

df['ds'] = pd.to_datetime(df['ds'])

print("🤖 Treinando a Inteligência Artificial do Prophet...")
# Inicializa o modelo adaptando as sazonalidades ao tipo de dado escolhido
model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=weekly_season, 
    daily_seasonality=False
)
model.fit(df)

print(f"🔮 Projetando o futuro ({periods_to_predict} períodos à frente)...")
future = model.make_future_dataframe(periods=periods_to_predict, freq=frequency)
forecast = model.predict(future)

print("📊 Gerando os gráficos...")
# 1. Gráfico Geral
fig1 = model.plot(forecast)
plt.title(f"Previsão de Gastos do Cartão de Crédito - Modo {GRANULARIDADE}")
plt.xlabel("Data")
plt.ylabel("Valor (R$)")
plt.grid(True, alpha=0.3)

# 2. Gráfico de Componentes (Tendência e Sazonalidades)
fig2 = model.plot_components(forecast)

print("\n🚀 Exibindo os gráficos na tela!")
plt.show()