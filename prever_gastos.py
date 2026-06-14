import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

print("⏳ Carregando os dados limpos...")
# 1. Carregar a base que limpamos
df = pd.read_csv("gastos_limpos.csv")
df['ds'] = pd.to_datetime(df['ds'])

print("🤖 Configurando e treinando a Inteligência Artificial...")
# 2. Inicializar o Prophet configurado para dados DIÁRIOS
# Ativamos a sazonalidade semanal (weekly) para ele entender o efeito "fim de semana"
model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=False
)
model.fit(df)

print("🔮 Projetando os gastos para os próximos 90 dias...")
# 3. Criar as datas futuras (periods=90 dias, freq='D' de diário)
future = model.make_future_dataframe(periods=90, freq='D')

# 4. Executar a previsão
forecast = model.predict(future)

print("📊 Gerando os gráficos de análise...")
# 5. Gráfico Principal: Histórico + Previsão
fig1 = model.plot(forecast)
plt.title("Previsão Diária de Gastos no Cartão de Crédito")
plt.xlabel("Data")
plt.ylabel("Gasto Diário Acumulado (R$)")
plt.grid(True, alpha=0.3)

# 6. Gráfico de Componentes: Tendência, Sazonalidade Anual e Semanal
fig2 = model.plot_components(forecast)

print("\n🚀 Pronto! Exibindo os gráficos na tela.")
plt.show()