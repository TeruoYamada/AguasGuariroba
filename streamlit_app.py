import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import io
import base64
import os
import tempfile
from datetime import datetime, timedelta
from PIL import Image
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração inicial do Streamlit
st.set_page_config(
    layout="wide",
    page_title="Águas Guariroba - Visualizador de Precipitação - MS",
    page_icon="🌧️"
)

# --- CONSTANTES E CONFIGURAÇÕES ---
CAMPOS_GRANDE_AREAS = {
    "Centro": [-20.4697, -54.6201],
    "Região Norte": [-20.4297, -54.6101],
    "Região Sul": [-20.5097, -54.6201],
    "Região Leste": [-20.4697, -54.5801],
    "Região Oeste": [-20.4697, -54.6601],
    "Região Centro-Norte": [-20.4397, -54.6301],
    "Região Centro-Sul": [-20.4997, -54.6301],
    "Região Nordeste": [-20.4397, -54.5901],
    "Região Noroeste": [-20.4397, -54.6501],
    "Região Sudeste": [-20.4997, -54.5901],
    "Região Sudoeste": [-20.4997, -54.6501]
}

# Mapeamento de variáveis para nomes do ERA5
ERA5_VARIABLES = {
    "total_precipitation": "tp",
    "large_scale_precipitation": "lsp",
    "convective_precipitation": "cp"
}

PRECIPITATION_VARIABLES = {
    "total_precipitation": "Precipitação Total (mm)",
    "large_scale_precipitation": "Precipitação de Grande Escala (mm)",
    "convective_precipitation": "Precipitação Convectiva (mm)"
}

COLORMAPS = ["Blues", "viridis", "plasma", "RdYlBu_r", "gist_earth"]

# Coordenadas aproximadas para o mapa de Campo Grande
CAMPO_GRANDE_BOUNDS = {
    'north': -20.35,
    'south': -20.60,
    'east': -54.50,
    'west': -54.75
}

# Definindo um diretório temporário para armazenar os arquivos baixados
TEMP_DIR = tempfile.gettempdir()


# --- FUNÇÕES DE INTERFACE E ESTILIZAÇÃO ---
def create_gradient_background():
    """Aplica um gradiente de fundo à aplicação."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to bottom, #1e3c72, #2a5298);
            color: white;
        }
        .stSidebar {
            background-color: rgba(30, 60, 114, 0.8);
        }
        .css-1adrfps {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .st-bw {
            color: white;
        }
        .st-cx {
            background-color: rgba(255, 255, 255, 0.1);
        }
        header {
            background-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def add_logo():
    """Adiciona o logo ao topo da aplicação."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/400x100?text=Águas+Guariroba", use_column_width=True)


# --- FUNÇÕES DE SIMULAÇÃO DE DADOS ---
def simulate_timeseries_data(start_date, end_date):
    """
    Simula dados de série temporal para precipitação.
    
    Args:
        start_date: Data inicial da série.
        end_date: Data final da série.
    
    Returns:
        DataFrame com colunas 'time' e 'precipitation'.
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        times = pd.date_range(start=start, end=end, freq='3H')
        
        # Usar vectorização numpy para melhor desempenho
        np.random.seed(42)  # Para reprodutibilidade
        
        # Horas do dia (0-23)
        hours = np.array([t.hour for t in times])
        
        # Fator baseado na hora do dia (mais chuva à tarde)
        hour_factors = np.where(
            (hours >= 6) & (hours <= 18),
            0.5 + 0.5 * np.sin(np.pi * (hours - 6) / 12),
            0.2
        )
        
        # Fatores aleatórios
        random_factors = np.random.exponential(1.0, size=len(times))
        
        # Eventos intensos aleatórios
        intense_events = 5 * (np.random.rand(len(times)) < 0.05)
        
        # Calcular precipitação
        precipitation = np.maximum(0, hour_factors * random_factors + intense_events)
        
        return pd.DataFrame({
            'time': times,
            'precipitation': precipitation
        })
    except Exception as e:
        logger.error(f"Erro ao simular dados de série temporal: {e}")
        return pd.DataFrame({'time': [], 'precipitation': []})


def simulate_daily_data(start_date, end_date):
    """
    Simula dados diários de precipitação.
    
    Args:
        start_date: Data inicial.
        end_date: Data final.
    
    Returns:
        DataFrame com colunas 'date' e 'precipitation'.
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        np.random.seed(42)
        
        # Meses (1-12)
        months = np.array([d.month for d in dates])
        
        # Fator sazonal (mais chuva no verão do hemisfério sul)
        season_factors = np.where(
            (months >= 11) | (months <= 3),
            1.5,
            0.7
        )
        
        # Fatores aleatórios
        random_factors = np.random.exponential(1.0, size=len(dates))
        
        # Calcular precipitação
        precipitation = np.maximum(0, season_factors * random_factors * 5)
        
        return pd.DataFrame({
            'date': dates,
            'precipitation': precipitation
        })
    except Exception as e:
        logger.error(f"Erro ao simular dados diários: {e}")
        return pd.DataFrame({'date': [], 'precipitation': []})


def simulate_forecast_data(end_date, forecast_days):
    """
    Simula dados de previsão de precipitação.
    
    Args:
        end_date: Data final dos dados históricos.
        forecast_days: Número de dias para previsão.
    
    Returns:
        DataFrame com colunas 'date' e 'precipitation'.
    """
    try:
        last_date = pd.to_datetime(end_date)
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        np.random.seed(43)  # Diferente da série histórica
        
        # Meses (1-12)
        months = np.array([d.month for d in dates])
        
        # Dias desde o início da previsão
        days_from_start = np.array([(d - dates[0]).days for d in dates])
        
        # Fator sazonal (mais chuva no verão do hemisfério sul)
        season_factors = np.where(
            (months >= 11) | (months <= 3),
            1.5,
            0.7
        )
        
        # Fator de dia (tendência decrescente ao longo do tempo)
        day_factors = np.maximum(0.1, 1 - days_from_start / forecast_days)
        
        # Fatores aleatórios
        random_factors = np.random.exponential(0.8, size=len(dates))
        
        # Calcular precipitação
        precipitation = np.maximum(0, season_factors * day_factors * random_factors * 5)
        
        return pd.DataFrame({
            'date': dates,
            'precipitation': precipitation
        })
    except Exception as e:
        logger.error(f"Erro ao simular dados de previsão: {e}")
        return pd.DataFrame({'date': [], 'precipitation': []})


def simulate_ml_forecast(regions, end_date, forecast_days):
    """
    Simula previsões de ML para cada região.
    
    Args:
        regions: Lista de nomes de regiões.
        end_date: Data final dos dados históricos.
        forecast_days: Número de dias para previsão.
    
    Returns:
        Dicionário com DataFrames de previsão por região.
    """
    try:
        ml_forecasts = {}
        
        for region in regions:
            last_date = pd.to_datetime(end_date)
            dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            
            # Usar hash do nome da região para seed
            region_seed = hash(region) % 100
            np.random.seed(region_seed)
            
            # Fator regional (norte mais chuvoso que sul)
            region_factor = 1.2 if "Norte" in region else 0.8 if "Sul" in region else 1.0
            
            # Meses (1-12)
            months = np.array([d.month for d in dates])
            
            # Fator sazonal (mais chuva no verão do hemisfério sul)
            season_factors = np.where(
                (months >= 11) | (months <= 3),
                1.5,
                0.7
            )
            
            # Fatores aleatórios
            random_factors = np.random.exponential(0.8, size=len(dates))
            
            # Calcular precipitação
            precipitation = np.maximum(0, region_factor * season_factors * random_factors * 5)
            
            ml_forecasts[region] = pd.DataFrame({
                'date': dates,
                'precipitation': precipitation,
                'region': region
            })
        
        return ml_forecasts
    except Exception as e:
        logger.error(f"Erro ao simular previsões ML: {e}")
        return {}


def generate_era5_data(params):
    """
    Gera dados simulados para substituir os dados do ERA5.
    
    Args:
        params: Dicionário com parâmetros de configuração.
    
    Returns:
        Dicionário com dados simulados.
    """
    try:
        st.warning("⚠️ Usando dados simulados para demonstração")
        
        # Usando processamento paralelo para acelerar a geração de dados
        results = {}
        
        # Simula série temporal para a área selecionada
        results['timeseries'] = simulate_timeseries_data(
            params['start_date'], 
            params['end_date']
        )
        
        # Simula dados diários
        results['daily'] = simulate_daily_data(
            params['start_date'], 
            params['end_date']
        )
        
        # Simula dados de previsão
        results['forecast'] = simulate_forecast_data(
            params['end_date'], 
            params['forecast_days']
        )
        
        # Simula dados para todas as regiões
        results['all_regions'] = {
            region: simulate_timeseries_data(params['start_date'], params['end_date'])
            for region in CAMPOS_GRANDE_AREAS.keys()
        }
        
        # Simula previsões ML
        results['ml_forecast'] = simulate_ml_forecast(
            list(CAMPOS_GRANDE_AREAS.keys()), 
            params['end_date'],
            params['forecast_days']
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"Erro ao gerar dados simulados: {e}")
        st.error(f"Erro ao gerar dados simulados: {str(e)}")
        return None


def try_import_optional_modules():
    """
    Tenta importar módulos opcionais e retorna status.
    
    Returns:
        Dicionário com status de importação dos módulos.
    """
    modules = {}
    
    # Lista de módulos para verificar
    module_list = [
        ('cdsapi', 'cdsapi'),
        ('xarray', 'xarray'),
        ('geopandas', 'geopandas'),
        ('cartopy.crs', 'cartopy'),
        ('sklearn.linear_model', 'sklearn')
    ]
    
    for module_name, key in module_list:
        try:
            __import__(module_name)
            modules[key] = True
        except ImportError:
            modules[key] = False
    
    return modules


# Função para criar um mapa simplificado quando o Cartopy não está disponível
def create_simple_map(data, title):
    """
    Cria um mapa simplificado sem o Cartopy.
    
    Args:
        data: Dados para visualização.
        title: Título do mapa.
    
    Returns:
        Figura do matplotlib.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Criar um gráfico de calor simples
        x = np.linspace(-54.75, -54.50, 20)
        y = np.linspace(-20.60, -20.35, 20)
        X, Y = np.meshgrid(x, y)
        
        # Gerar dados aleatórios mais consistentes com o título
        np.random.seed(hash(title) % 100)  # Seed baseado no título
        Z = np.random.rand(20, 20) * 10
        
        # Suavizar dados para mapas mais realistas
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=1.0)
        
        # Criar o mapa de calor
        c = ax.pcolormesh(X, Y, Z, cmap='Blues', alpha=0.7)
        
        # Adicionar pontos representando as regiões
        for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            ax.plot(lon, lat, 'ro', markersize=4)
            ax.text(lon + 0.01, lat + 0.01, region, fontsize=8, 
                   fontweight='bold', color='black', 
                   path_effects=[plt.patheffects.withStroke(linewidth=2, foreground='white')])
        
        # Formatar o gráfico
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        
        # Adicionar colorbar
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label('Precipitação (mm)')
        
        # Melhorar o visual
        ax.set_facecolor('#f0f0f0')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Erro ao criar mapa simplificado: {e}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"Erro ao criar mapa: {str(e)}", 
                ha='center', va='center', fontsize=12)
        return fig


# Função principal da aplicação
def main():
    """Função principal que executa a aplicação Streamlit."""
    try:
        # Verificar módulos disponíveis
        modules = try_import_optional_modules()
        
        # Aplicando estilização
        create_gradient_background()
        
        # Cabeçalho
        add_logo()
        st.title("📊 Visualizador de Precipitação - Campo Grande, MS")
        st.markdown("### Sistema de Monitoramento e Previsão de Chuvas")
        
        # Sidebar para configurações
        with st.sidebar:
            st.header("⚙️ Configurações")
            
            # Abas para organizar as configurações
            config_tab1, config_tab2, config_tab3 = st.tabs(["📍 Local", "📅 Período", "🔧 Avançado"])
            
            with config_tab1:
                # Seleção de região
                area = st.selectbox(
                    "Selecione a região",
                    list(CAMPOS_GRANDE_AREAS.keys()),
                    index=0
                )
                lat_center, lon_center = CAMPOS_GRANDE_AREAS.get(area, (-20.4697, -54.6201))
                
                # Visualização do mapa
                map_width = st.slider("Área de Visualização (graus)", 0.1, 2.0, 0.3, 0.1)
                show_shapefile = st.checkbox("Mostrar Área Urbana", value=True)
                satellite_background = st.checkbox("Usar Imagem de Satélite", value=True)
            
            with config_tab2:
                # Período de análise
                st.subheader("Período de Análise")
                today = datetime.today()
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Início", today - timedelta(days=7))
                with col2:
                    end_date = st.date_input("Fim", today + timedelta(days=5))
                
                # Horários
                st.subheader("Horários")
                all_hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
                col1, col2 = st.columns(2)
                with col1:
                    start_hour = st.selectbox("Hora Inicial", all_hours)
                with col2:
                    end_hour = st.selectbox("Hora Final", all_hours, index=len(all_hours)-1)

                # Variável de precipitação
                precip_var = st.selectbox(
                    "Variável de Precipitação", 
                    list(PRECIPITATION_VARIABLES.keys()),
                    format_func=lambda x: PRECIPITATION_VARIABLES[x]
                )
                
                # Horizonte de previsão
                forecast_days = st.slider("Horizonte de Previsão (dias)", 1, 14, 7)
                
            with config_tab3:
                # Configurações avançadas
                st.subheader("Configurações Avançadas")
                
                animation_speed = st.slider("Velocidade Animação (ms)", 200, 1000, 500)
                colormap = st.selectbox("Paleta de Cores", COLORMAPS)
                
                product_type = st.radio("Tipo de Produto", ["reanalysis", "ensemble_mean"])
                
                # Verificar se o sklearn está disponível para mostrar opções de modelo
                if modules['sklearn']:
                    ml_model = st.selectbox("Modelo de Previsão", 
                                          ["RandomForest", "GradientBoosting", "LinearRegression"])
                else:
                    ml_model = "LinearRegression"  # Valor padrão
                    st.warning("Scikit-learn não está instalado. Usando modelo padrão.")
                
                probability_threshold = st.slider("Limiar de Probabilidade (%)", 0, 100, 30)
        
        # Botão para atualizar dados
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            update_button = st.button("🔄 Atualizar Dados", use_container_width=True)
        
        # Verificar módulos necessários e mostrar avisos
        missing_modules = [k for k, v in modules.items() if not v]
        if missing_modules:
            st.warning(f"⚠️ Os seguintes módulos não estão instalados: {', '.join(missing_modules)}. Alguns recursos podem não funcionar.")
        
        # Organizar a exibição dos dados em abas
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Série Temporal", "🗺️ Mapas", "🔮 Previsões", "📊 Análise Regional"])
        
        # Preparar parâmetros para processamento
        params = {
            'area': area,
            'lat_center': lat_center,
            'lon_center': lon_center,
            'start_date': start_date,
            'end_date': end_date,
            'start_hour': int(start_hour.split(':')[0]),
            'end_hour': int(end_hour.split(':')[0]),
            'precip_var': precip_var,
            'map_width': map_width,
            'animation_speed': animation_speed,
            'colormap': colormap,
            'product_type': product_type,
            'forecast_days': forecast_days,
            'ml_model': ml_model,
            'probability_threshold': probability_threshold,
            'show_shapefile': show_shapefile,
            'satellite_background': satellite_background
        }
        
        # Simular dados para demonstração
        if 'data' not in st.session_state or update_button:
            with st.spinner("⌛ Carregando dados..."):
                # Para demonstração, usar dados simulados
                st.session_state.data = generate_era5_data(params)
        
        data = st.session_state.data
        
        # Tab 1: Série Temporal
        with tab1:
            st.header(f"Série Temporal de Precipitação - {area}")
            
            if data and 'timeseries' in data and not data['timeseries'].empty:
                # Gráfico de série temporal
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(data['timeseries']['time'], data['timeseries']['precipitation'],
                    width=0.02, alpha=0.7, color='#1e88e5', label='Precipitação a cada 3h')
                
                # Formatar eixos
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                plt.xticks(rotation=45)
                ax.set_ylabel('Precipitação (mm)', fontsize=12)
                ax.set_title(f'Precipitação em {area} - {start_date.strftime("%d/%m/%Y")} a {end_date.strftime("%d/%m/%Y")}', 
                            fontsize=14)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Estatísticas diárias
                st.subheader("Estatísticas Diárias")
                
                if 'daily' in data and not data['daily'].empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Precipitação Total", f"{data['daily']['precipitation'].sum():.1f} mm")
                    with col2:
                        st.metric("Precipitação Média Diária", f"{data['daily']['precipitation'].mean():.1f} mm/dia")
                    with col3:
                        st.metric("Dias com Chuva", f"{(data['daily']['precipitation'] > 0.1).sum()} dias")
                    
                    # Tabela de dados diários
                    st.subheader("Dados Diários")
                    display_df = data['daily'].copy()
                    
                    # Converter datas para string no formato brasileiro
                    if 'date' in display_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(display_df['date']):
                            display_df['date'] = display_df['date'].dt.strftime('%d/%m/%Y')
                    
                    display_df.columns = ['Data', 'Precipitação (mm)']
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("Dados diários não estão disponíveis.")
            else:
                st.warning("Dados de série temporal não estão disponíveis.")
        
        # Tab 2: Mapas
        with tab2:
            st.header("Visualização Espacial da Precipitação")
            
            if data and 'timeseries' in data and not data['timeseries'].empty:
                # Seletor de tempo para o mapa
                timestamps = data['timeseries']['time'].tolist()
                selected_time = st.select_slider(
                    "Selecione um Momento:",
                    options=timestamps,
                    format_func=lambda x: x.strftime('%d/%m/%Y %H:%M')
                )
                
                # Criar duas colunas para os mapas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Mapa de Precipitação")
                    
                    if modules['cartopy']:
                        # Aqui normalmente chamaríamos create_precip_map
                        st.image("https://via.placeholder.com/800x600?text=Mapa+de+Precipitação", use_column_width=True)
                    else:
                        # Usar mapa simplificado
                        fig = create_simple_map(data, f"Precipitação em {selected_time.strftime('%d/%m/%Y %H:%M')}")
                        st.pyplot(fig)
                    
                with col2:
                    st.subheader("Probabilidade de Chuva")
                    
                    if modules['cartopy']:
                        # Aqui normalmente chamaríamos create_probability_map
                        st.image("https://via.placeholder.com/800x600?text=Mapa+de+Probabilidade", use_column_width=True)
                    else:
                        # Usar mapa simplificado
                        fig = create_simple_map(data, "Probabilidade de Precipitação")
                        st.pyplot(fig)
                
                # Animação
                st.subheader("Animação da Precipitação")
                animation_placeholder = st.empty()
                animation_placeholder.image("https://via.placeholder.com/800x600?text=Animação+(GIF)", use_column_width=True)
                
                # Opção para download da animação
                st.download_button(
                    label="⬇️ Download da Animação",
                    data=io.BytesIO(b"Placeholder para o GIF real"),
                    file_name="precipitacao_animacao.gif",
                    mime="image/gif"
                )
            else:
                st.warning("Dados para mapas não estão disponíveis.")
        
        # Tab 3: Previsões
        with tab3:
            st.header("Previsão de Precipitação")
            
            # Escolha do método de previsão
            forecast_method = st.radio(
                "Método de Previsão:",
                ["Modelo Linear", "Machine Learning", "Média dos Métodos"],
                horizontal=True
            )
            
            if data and 'daily' in data and not data['daily'].empty and 'forecast' in data and not data['forecast'].empty:
                # Gráfico de previsão
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Dados históricos
                historical_dates = data['daily']['date'].tolist()
                historical_precip = data['daily']['precipitation'].tolist()
                ax.bar(historical_dates, historical_precip, width=0.6, alpha=0.7, color='#1e88e5', label='Histórico')
                
                # Dados de previsão
                forecast_dates = data['forecast']['date'].tolist()
                forecast_precip = data['forecast']['precipitation'].tolist()
                ax.bar(forecast_dates, forecast_precip, width=0.6, alpha=0.7, color='#ff9800', label='Previsão')
                
                # Formatar eixos
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
                plt.xticks(rotation=45)
                ax.set_ylabel('Precipitação (mm/dia)', fontsize=12)
                ax.set_title(f'Previsão de Precipitação para {area} - Próximos {forecast_days} dias', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Detalhes da previsão
                st.subheader("Detalhes da Previsão")
                
                # Estatísticas de previsão
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precipitação Total Prevista", f"{data['forecast']['precipitation'].sum():.1f} mm")
                with col2:
                    st.metric("Precipitação Máxima Diária", f"{data['forecast']['precipitation'].max():.1f} mm")
                with col3:
                    st.metric("Dias com Chuva Previstos", f"{(data['forecast']['precipitation'] > 0.1).sum()} dias")
                
                # Tabela de previsão
                st.subheader("Tabela de Previsão")
                display_forecast = data['forecast'].copy()
                
                # Converter datas para string no formato brasileiro
                if 'date' in display_forecast.columns:
                    if pd.api.types.is_datetime64_any_dtype(display_forecast['date']):
                        display_forecast['date'] = display_forecast['date'].dt.strftime('%d/%m/%Y')
                
                display_forecast.columns = ['Data', 'Precipitação Prevista (mm)']
                st.dataframe(display_forecast, use_container_width=True)
                
                # Alertas de chuva
                st.subheader("⚠️ Alertas de Precipitação")
                
                # Condição de chuva forte
                heavy_rain_days = data['forecast'][data['forecast']['precipitation'] > 10]
                if not heavy_rain_days.empty:
                    st.warning(f"Possibilidade de chuva forte nos dias: {', '.join(heavy_rain_days['date'].dt.strftime('%d/%m/%Y').tolist())}")
                else:
                    st.success("Não há previsão de chuvas fortes para o período selecionado.")
                
            else:
                st.warning("Dados de previsão não estão disponíveis.")
        
        # Tab 4: Análise Regional
        with tab4:
            st.header("Análise Regional de Precipitação")
            
            if data and 'all_regions' in data and data['all_regions']:
                # Seleção de regiões para comparação
                all_regions = list(CAMPOS_GRANDE_AREAS.keys())
                selected_regions = st.multiselect(
                    "Selecione regiões para comparar:",
                    all_regions,
                    default=all_regions[:3]
                )
                
                if selected_regions:
                    # Gráfico de comparação regional
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for region in selected_regions:
                        if region in data['all_regions']:
                            region_data = data['all_regions'][region]
                            # Calcular média diária para simplificar a visualização
                            region_data['date'] = pd.to_datetime(region_data['time']).dt.date
                            daily_avg = region_data.groupby('date')['precipitation'].mean().reset_index()
                            
                            # Converter para datetime para usar no gráfico
                            daily_avg['date'] = pd.to_datetime(daily_avg['date'])
                            
                            ax.plot(daily_avg['date'], daily_avg['precipitation'], 
                                  marker='o', markersize=3, linewidth=2, label=region)
                    
                    # Formatar eixos
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
                    plt.xticks(rotation=45)
                    ax.set_ylabel('Precipitação Média (mm/dia)', fontsize=12)
                    ax.set_title('Comparação de Precipitação por Região', fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Mapa de calor por região
                    st.subheader("Mapa de Calor por Região")
                    
                    # Preparar dados para o mapa de calor
                    heat_data = []
                    for region in all_regions:
                        if region in data['all_regions']:
                            region_data = data['all_regions'][region]
                            region_data['date'] = pd.to_datetime(region_data['time']).dt.date
                            daily_total = region_data.groupby('date')['precipitation'].sum().reset_index()
                            
                            for _, row in daily_total.iterrows():
                                heat_data.append({
                                    'region': region,
                                    'date': row['date'],
                                    'precipitation': row['precipitation']
                                })
                    
                    if heat_data:
                        heat_df = pd.DataFrame(heat_data)
                        
                        # Criar pivot table para o mapa de calor
                        pivot_table = heat_df.pivot_table(
                            values='precipitation', 
                            index='region', 
                            columns='date', 
                            aggfunc='sum'
                        )
                        
                        # Plotar mapa de calor
                        fig, ax = plt.subplots(figsize=(14, 8))
                        c = ax.pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values, 
                                       cmap=colormap, alpha=0.8)
                        
                        # Formatar eixos
                        ax.set_yticks(np.arange(0.5, len(pivot_table.index), 1), pivot_table.index)
                        ax.set_xticks(pivot_table.columns[::2])  # Mostrar apenas datas alternadas
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
                        plt.xticks(rotation=45)
                        
                        # Adicionar barra de cores
                        cbar = plt.colorbar(c, ax=ax)
                        cbar.set_label('Precipitação Total (mm)', fontsize=12)
                        
                        ax.set_title('Distribuição de Precipitação por Região', fontsize=14)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    else:
                        st.warning("Dados insuficientes para o mapa de calor.")
                    
                    # Alertas regionais
                    st.subheader("⚠️ Alertas Regionais")
                    
                    # Identificar regiões com precipitação acima da média
                    if 'ml_forecast' in data and data['ml_forecast']:
                        high_precip_regions = []
                        threshold = 15  # Limiar para alerta (mm)
                        
                        for region, forecast_data in data['ml_forecast'].items():
                            total_precip = forecast_data['precipitation'].sum()
                            if total_precip > threshold:
                                high_precip_regions.append((region, total_precip))
                        
                        if high_precip_regions:
                            high_precip_regions.sort(key=lambda x: x[1], reverse=True)
                            
                            st.warning("Regiões com previsão de alta precipitação para os próximos dias:")
                            
                            for region, precip in high_precip_regions:
                                st.markdown(f"- **{region}**: {precip:.1f} mm")
                        else:
                            st.success("Não há previsão de precipitação alta para nenhuma região.")
                    else:
                        st.warning("Dados de previsão regional não estão disponíveis.")
                else:
                    st.warning("Selecione pelo menos uma região para comparação.")
            else:
                st.warning("Dados regionais não estão disponíveis.")
        
        # Rodapé da aplicação
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📝 Exportar Relatório")
            export_format = st.selectbox("Formato:", ["PDF", "Excel", "CSV"])
            
            st.download_button(
                label="⬇️ Exportar Dados",
                data=io.BytesIO(b"Simulação de exportação de dados"),
                file_name=f"precipitacao_{area}_{start_date}_{end_date}.{export_format.lower()}",
                mime="application/octet-stream"
            )
        
        with col2:
            st.markdown("### 📊 Estatísticas Gerais")
            if data and 'daily' in data and not data['daily'].empty:
                # Calcular estatísticas adicionais
                percentile_90 = np.percentile(data['daily']['precipitation'], 90)
                
                st.markdown(f"""
                - **Média Mensal Histórica**: 150 mm
                - **Percentil 90%**: {percentile_90:.1f} mm
                - **Dias Sem Chuva**: {(data['daily']['precipitation'] < 0.1).sum()} dias
                - **Dias com Precipitação Alta**: {(data['daily']['precipitation'] > 20).sum()} dias
                """)
            else:
                st.warning("Estatísticas não disponíveis.")
        
        with col3:
            st.markdown("### ⚙️ Sobre o Sistema")
            st.markdown("""
            **Águas Guariroba - Sistema de Monitoramento e Previsão**
            
            Versão: 1.0.0 (Demonstração)
            
            Dados baseados em simulação para demonstração.
            """)
        
        # Avisos finais
        st.markdown("---")
        st.markdown("""
        **Observações**: Este é um sistema de demonstração. Em um ambiente de produção, os dados seriam obtidos de fontes oficiais
        como INMET, CEMADEN, ou diretamente do ERA5 via API CDS.
        """)
        
    except Exception as e:
        st.error(f"Erro na aplicação: {str(e)}")
        logger.exception("Erro na aplicação")


# Executar a aplicação
if __name__ == "__main__":
    main()
