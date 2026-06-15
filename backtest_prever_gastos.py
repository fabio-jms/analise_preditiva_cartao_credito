import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics

# Colocamos todo o código de execução dentro desta proteção (essencial no Windows)
if __name__ == '__main__':

    # =====================================================================
    # CONFIGURAÇÃO DE GRANULARIDADE
    # Mude para 'MENSAL' ou 'DIARIO' dependendo do que quer analisar agora
    GRANULARIDADE = 'MENSAL' 
    # =====================================================================

    print(f"⏳ Carregando os dados no modo: {GRANULARIDADE}...")

    if GRANULARIDADE == 'MENSAL':
        df = pd.read_csv("gastos_limpos_mensal.csv")
        periods_to_predict = 12  
        frequency = 'MS'         
        weekly_season = False    
        
        # ✨ Ajuste fino para a validação mensal funcionar perfeitamente:
        horizonte = '150 days'   # Prever 5 meses à frente em cada teste (mais seguro para dados mensais)
        periodo = '90 days'      # Faz um novo teste a cada 3 meses de histórico
        treino_inicial = '1095 days' # Começa o primeiro teste após 3 anos de histórico (sobra base!)

    elif GRANULARIDADE == 'DIARIO':
        df = pd.read_csv("gastos_limpos_diario.csv")
        periods_to_predict = 90  
        frequency = 'D'          
        weekly_season = True     
        
        # Configuração da Validação Cruzada Diária
        horizonte = '90 days'    
        periodo = '180 days'     
        treino_inicial = '2190 days' 

    else:
        print("❌ Modo inválido! Escolha 'MENSAL' ou 'DIARIO'.")
        exit()

    df['ds'] = pd.to_datetime(df['ds'])

    print("🤖 Treinando a Inteligência Artificial...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=weekly_season, daily_seasonality=False)
    model.add_country_holidays(country_name='BR')
    model.fit(df)

    # =====================================================================
    # VALIDAÇÃO CRUZADA (CORRIGIDA E ADAPTADA PARA REAIS)
    # =====================================================================
    print("\n🔄 Iniciando Validação Cruzada (Simulando previsões no passado)...")
    print("Calculando o erro médio real em Reais (R$)...")

    try:
        # Forçamos parâmetros mais flexíveis para o mensal não falhar
        if GRANULARIDADE == 'MENSAL':
            df_cv = cross_validation(
                model, 
                initial='1460 days', # 4 anos de treino inicial
                period='180 days',   # Testa a cada 6 meses
                horizon='180 days',  # Prevê 6 meses à frente
                parallel=None
            )
        else: # DIÁRIO
            df_cv = cross_validation(
                model, 
                initial='2190 days', 
                period='365 days', 
                horizon='90 days',  
                parallel=None
            )

        # Em vez de usar a função performance_metrics que exige o MAPE,
        # calculamos a diferença absoluta real em Reais (R$)
        df_cv['erro_absoluto'] = (df_cv['y'] - df_cv['yhat']).abs()
        erro_medio_reais = df_cv['erro_absoluto'].mean()

        print("\n==================================================")
        print("       🎯 DIAGNÓSTICO DE PRECISÃO DA IA          ")
        print("==================================================")
        if GRANULARIDADE == 'MENSAL':
            print(f"🔹 Erro Médio por Fatura Mensal: R$ {erro_medio_reais:.2f}")
            print("  (Significa que a IA erra o valor da sua fatura para mais")
            print(f"   ou para menos por cerca de R$ {erro_medio_reais:.2f} na média)")
        else:
            print(f"🔹 Erro Médio por Gasto Diário: R$ {erro_medio_reais:.2f}")
            print("  (Na flutuação de cada dia, a IA erra a estimativa")
            print(f"   por cerca de R$ {erro_medio_reais:.2f})")
        print("==================================================\n")
        
        titulo_grafico = f"Previsão de Gastos - Modo {GRANULARIDADE}\n(Erro Médio: R$ {erro_medio_reais:.2f})"

    except Exception as e:
        print(f"\n⚠️ Não foi possível calcular a validação cruzada: {e}")
        titulo_grafico = f"Previsão de Gastos - Modo {GRANULARIDADE}"

    # =====================================================================
    # PROJEÇÃO FUTURA REAL
    # =====================================================================
    print(f"🔮 Projetando o futuro real ({periods_to_predict} períodos à frente)...")
    future = model.make_future_dataframe(periods=periods_to_predict, freq=frequency)
    forecast = model.predict(future)

    print("📊 Gerando os gráficos...")
    fig1 = model.plot(forecast)
    add_changepoints_to_plot(fig1.gca(), model, forecast)
    plt.title(titulo_grafico)
    plt.xlabel("Data")
    plt.ylabel("Valor (R$)")
    plt.grid(True, alpha=0.3)

    plt.show()

    