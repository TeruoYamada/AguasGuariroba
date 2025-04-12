import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import io
import base64
from PIL import Image
from datetime import datetime, timedelta

# Configuração inicial
st.set_page_config(
    layout="wide",
    page_title="Águas Guariroba - Visualizador de Precipitação - MS",
    page_icon="🌧️"
)

# Funções auxiliares para estilização
def create_gradient_background():
    # Código CSS para gradiente de fundo
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
    # Adicionar logo (placeholder)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/400x100?text=Águas+Guariroba", use_column_width=True)

# Criando a interface
def main():
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
            ml_model = st.selectbox("Modelo de Previsão", 
                                  ["RandomForest", "GradientBoosting", "LinearRegression"])
            
            probability_threshold = st.slider("Limiar de Probabilidade (%)", 0, 100, 30)
    
    # Botão para atualizar dados
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        update_button = st.button("🔄 Atualizar Dados", use_container_width=True)

    # Organizar a exibição dos dados em abas
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Série Temporal", "🗺️ Mapas", "🔮 Previsões", "📊 Análise Regional"])
    
    # Simular dados para demonstração
    if 'data' not in st.session_state or update_button:
        with st.spinner("⌛ Carregando dados..."):
            # Normalmente aqui chamaríamos a função download_era5_data e process_precipitation_data
            # Para demonstração, vamos gerar dados simulados
            st.session_state.data = {
                'timeseries': simulate_timeseries_data(start_date, end_date),
                'daily': simulate_daily_data(start_date, end_date),
                'forecast': simulate_forecast_data(end_date, forecast_days),
                'ml_forecast': simulate_ml_forecast(list(CAMPOS_GRANDE_AREAS.keys()), end_date, forecast_days),
                'all_regions': {region: simulate_timeseries_data(start_date, end_date) 
                              for region in CAMPOS_GRANDE_AREAS.keys()}
            }
    
    # Tab 1: Série Temporal
    with tab1:
        st.header(f"Série Temporal de Precipitação - {area}")
        
        data = st.session_state.data
        
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
        display_df['date'] = display_df['date'].dt.strftime('%d/%m/%Y')
        display_df.columns = ['Data', 'Precipitação (mm)']
        st.dataframe(display_df, use_container_width=True)
    
    # Tab 2: Mapas
    with tab2:
        st.header("Visualização Espacial da Precipitação")
        
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
            # Aqui normalmente chamaríamos create_precip_map
            # Por ora vamos mostrar uma imagem placeholder
            st.image("https://via.placeholder.com/800x600?text=Mapa+de+Precipitação", use_column_width=True)
            
        with col2:
            st.subheader("Probabilidade de Chuva")
            # Aqui normalmente chamaríamos a função que gera o mapa de probabilidade
            st.image("https://via.placeholder.com/800x600?text=Mapa+de+Probabilidade", use_column_width=True)
        
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
    
    # Tab 3: Previsões
    with tab3:
        st.header("Previsão de Precipitação")
        
        # Escolha do modelo de previsão
        forecast_method = st.radio(
            "Método de Previsão:",
            ["Modelo Linear", "Machine Learning", "Média dos Métodos"],
            horizontal=True
        )
        
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
        st.subheader("Dados de Previsão")
        forecast_display = data['forecast'].copy()
        forecast_display['date'] = forecast_display['date'].dt.strftime('%d/%m/%Y')
        forecast_display = forecast_display[['date', 'precipitation']]
        forecast_display.columns = ['Data', 'Precipitação Prevista (mm)']
        st.dataframe(forecast_display, use_container_width=True)
    
    # Tab 4: Análise Regional
    with tab4:
        st.header("Comparação Entre Regiões")
        
        # Selecionar regiões para comparação
        selected_regions = st.multiselect(
            "Selecione as regiões para comparar:",
            list(CAMPOS_GRANDE_AREAS.keys()),
            default=["Centro", "Região Norte", "Região Sul"]
        )
        
        if selected_regions:
            # Gráfico de comparação entre regiões
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for region in selected_regions:
                region_data = data['all_regions'][region]
                region_daily = region_data.groupby('date')['precipitation'].sum().reset_index()
                ax.plot(region_daily['date'], region_daily['precipitation'], linewidth=2, label=region)
            
            # Formatar eixos
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.xticks(rotation=45)
            ax.set_ylabel('Precipitação (mm/dia)', fontsize=12)
            ax.set_title('Comparação de Precipitação Entre Regiões', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Mapa de calor regional
            st.subheader("Mapa de Calor Regional")
            st.image("https://via.placeholder.com/800x600?text=Mapa+de+Calor+Regional", use_column_width=True)
            
            # Tabela comparativa
            st.subheader("Tabela Comparativa")
            
            # Criar tabela comparativa
            comparison_data = []
            for region in selected_regions:
                region_data = data['all_regions'][region]
                region_daily = region_data.groupby('date')['precipitation'].sum().reset_index()
                
                region_forecast = data['ml_forecast'].get(region, pd.DataFrame())
                if not region_forecast.empty:
                    forecast_sum = region_forecast['precipitation'].sum()
                    forecast_max = region_forecast['precipitation'].max()
                else:
                    forecast_sum = "-"
                    forecast_max = "-"
                
                comparison_data.append({
                    "Região": region,
                    "Precipitação Total (mm)": round(region_daily['precipitation'].sum(), 1),
                    "Média Diária (mm)": round(region_daily['precipitation'].mean(), 1),
                    "Máxima Diária (mm)": round(region_daily['precipitation'].max(), 1),
                    "Previsão Total (mm)": forecast_sum if isinstance(forecast_sum, float) else forecast_sum,
                    "Previsão Máxima (mm)": forecast_max if isinstance(forecast_max, float) else forecast_max
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    # Rodapé
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Desenvolvedora:** Águas Guariroba")
    with col2:
        st.markdown("**Fonte de Dados:** ERA5 - Climate Data Store")
    with col3:
        st.markdown("**Última Atualização:** " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# --- DADOS SIMULADOS PARA DEMONSTRAÇÃO ---
def simulate_timeseries_data(start_date, end_date):
    """Simula dados de série temporal"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    times = pd.date_range(start=start, end=end, freq='3H')
    
    # Gerar dados aleatórios com alguns padrões
    np.random.seed(42)  # Para reprodutibilidade
    
    # Gerar precipitação baseada em padrões diários e aleatórios
    precipitation = []
    for t in times:
        hour = t.hour
        # Mais chuva à tarde
        hour_factor = 0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.2
        # Padrão aleatório
        random_factor = np.random.exponential(1.0)
        # Alguns eventos de chuva intensa
        intense = 5 * int(np.random.rand() < 0.05)
        
        value = max(0, hour_factor * random_factor + intense)
        precipitation.append(value)
    
    return pd.DataFrame({
        'time': times,
        'precipitation': precipitation
    })

def simulate_daily_data(start_date, end_date):
    """Simula dados diários"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    np.random.seed(42)
    precipitation = []
    
    for d in dates:
        # Mais chuva no verão (considerando hemisfério sul)
        month = d.month
        season_factor = 1.5 if (month >= 11 or month <= 3) else 0.7
        # Padrão aleatório
        random_factor = np.random.exponential(1.0)
        
        value = max(0, season_factor * random_factor * 5)
        precipitation.append(value)
    
    return pd.DataFrame({
        'date': dates,
        'precipitation': precipitation
    })

def simulate_forecast_data(end_date, forecast_days):
    """Simula dados de previsão"""
    last_date = pd.to_datetime(end_date)
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    
    np.random.seed(43)  # Diferente da série histórica
    precipitation = []
    
    for d in dates:
        # Mais chuva no verão (considerando hemisfério sul)
        month = d.month
        season_factor = 1.5 if (month >= 11 or month <= 3) else 0.7
        # Padrão aleatório com tendência de diminuição
        day_factor = max(0.1, 1 - (d - dates[0]).days / forecast_days)
        random_factor = np.random.exponential(0.8)
        
        value = max(0, season_factor * day_factor * random_factor * 5)
        precipitation.append(value)
    
    return pd.DataFrame({
        'date': dates,
        'precipitation': precipitation
    })

def simulate_ml_forecast(regions, end_date, forecast_days):
    """Simula previsões de ML para cada região"""
    ml_forecasts = {}
    
    for region in regions:
        last_date = pd.to_datetime(end_date)
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        np.random.seed(hash(region) % 100)  # Semente diferente para cada região
        precipitation = []
        
        for d in dates:
            # Variar por região (norte mais chuvoso que sul, etc.)
            region_factor = 1.2 if "Norte" in region else 0.8 if "Sul" in region else 1.0
            # Mais chuva no verão
            month = d.month
            season_factor = 1.5 if (month >= 11 or month <= 3) else 0.7
            # Padrão aleatório
            random_factor = np.random.exponential(0.8)
            
            value = max(0, region_factor * season_factor * random_factor * 5)
            precipitation.append(value)
        
        ml_forecasts[region] = pd.DataFrame({
            'date': dates,
            'precipitation': precipitation,
            'region': region
        })
    
    return ml_forecasts

# Constantes necessárias para o funcionamento (simplificadas)
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

PRECIPITATION_VARIABLES = {
    "total_precipitation": "Precipitação Total (mm)",
    "large_scale_precipitation": "Precipitação de Grande Escala (mm)",
    "convective_precipitation": "Precipitação Convectiva (mm)"
}

COLORMAPS = ["Blues", "viridis", "plasma", "RdYlBu_r", "gist_earth"]

# Definindo um diretório temporário para armazenar os arquivos baixados
TEMP_DIR = tempfile.gettempdir()

# ✅ Autenticação CDS (ERA5)
@st.cache_resource
def get_cds_client():
    try:
        return cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=st.secrets["cds"]["key"]
        )
    except Exception as e:
        st.error(f"❌ Erro ao conectar ao Climate Data Store: {str(e)}")
        return None

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

# Coordenadas da área urbana de Campo Grande
CAMPO_GRANDE_SHAPE = {
    'bounding_box': {
        'north': -20.35,
        'south': -20.60,
        'east': -54.50,
        'west': -54.75
    }
}

# Shapefile de Campo Grande (definido inline para evitar dependências de arquivo)
@st.cache_data
def get_campo_grande_shapefile():
    # URL do shapefile de Campo Grande ou gerar um aproximado
    try:
        # Tentar carregar o shapefile do Github ou outra fonte
        url = "https://raw.githubusercontent.com/CampoGrandeData/GIS/main/campo_grande_urban_area.geojson"
        campo_grande_gdf = gpd.read_file(url)
        return campo_grande_gdf
    except Exception as e:
        logger.warning(f"Erro ao carregar shapefile externo: {str(e)}")
        
        # Criar um polígono simples baseado no bounding box como alternativa
        from shapely.geometry import Polygon
        
        bbox = CAMPO_GRANDE_BOUNDS
        polygon = Polygon([
            (bbox['west'], bbox['north']),
            (bbox['east'], bbox['north']),
            (bbox['east'], bbox['south']),
            (bbox['west'], bbox['south'])
        ])
        
        # Criar GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        return gdf

# --- FUNÇÕES AUXILIARES ---
def setup_sidebar():
    """Configura a barra lateral com parâmetros de entrada"""
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção de região
    area = st.sidebar.selectbox("Selecione a região", list(CAMPOS_GRANDE_AREAS.keys()))
    lat_center, lon_center = CAMPOS_GRANDE_AREAS[area]
    
    # Período de análise
    st.sidebar.subheader("Período de Análise")
    today = datetime.today()
    start_date = st.sidebar.date_input("Data Início", today - timedelta(days=7))
    end_date = st.sidebar.date_input("Data Fim", today + timedelta(days=5))
    
    # Horários
    all_hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
    start_hour = st.sidebar.selectbox("Hora Inicial", all_hours)
    end_hour = st.sidebar.selectbox("Hora Final", all_hours, index=len(all_hours)-1)
    
    # Variável de precipitação
    precip_var = st.sidebar.selectbox(
        "Variável", 
        list(PRECIPITATION_VARIABLES.keys()),
        format_func=lambda x: PRECIPITATION_VARIABLES[x]
    )
    
    # Forecast horizon
    forecast_days = st.sidebar.slider("Horizonte de Previsão (dias)", 1, 14, 7)
    
    # Opções avançadas
    with st.sidebar.expander("Configurações Avançadas"):
        map_width = st.slider("Largura do Mapa (graus)", 0.1, 2.0, 0.3, 0.1)
        animation_speed = st.slider("Velocidade Animação (ms)", 200, 1000, 500)
        colormap = st.selectbox("Paleta de Cores", COLORMAPS)
        product_type = st.radio("Tipo de Produto", ["reanalysis", "ensemble_mean"])
        ml_model = st.selectbox("Modelo de Previsão", ["RandomForest", "GradientBoosting", "LinearRegression"])
        probability_threshold = st.slider("Limiar de Probabilidade (%)", 0, 100, 30)
        show_shapefile = st.checkbox("Mostrar Área Urbana", value=True)
        satellite_background = st.checkbox("Usar Imagem de Satélite", value=True)
    
    return {
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

def download_era5_data(params, client):
    """Baixa dados do ERA5 com tratamento robusto de variáveis"""
    try:
        # Verificar se as datas são válidas
        if params['start_date'] > params['end_date']:
            st.error("❌ Data de início maior que data de fim")
            return None
            
        # Se não temos cliente CDS, simular dados para demonstração
        if client is None:
            st.warning("⚠️ Usando dados simulados para demonstração (CDS não disponível)")
            return simulate_era5_data(params)

        # Obter nome correto da variável no ERA5
        era5_var = ERA5_VARIABLES.get(params['precip_var'])
        if not era5_var:
            st.error(f"❌ Variável {params['precip_var']} não mapeada")
            return None

        # Usar um arquivo temporário em vez de salvar diretamente no diretório atual
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False, dir=TEMP_DIR) as temp_file:
            filename = temp_file.name
        
        area = [
            params['lat_center'] + params['map_width'],
            params['lon_center'] - params['map_width'],
            params['lat_center'] - params['map_width'],
            params['lon_center'] + params['map_width']
        ]

        # Converter objetos date para strings no formato esperado pelo ERA5
        start_date_str = params['start_date'].strftime('%Y-%m-%d')
        end_date_str = params['end_date'].strftime('%Y-%m-%d')
        
        date_range = pd.date_range(start_date_str, end_date_str)
        time_list = [f"{h:02d}:00" for h in range(params['start_hour'], params['end_hour'] + 1, 3)]
        
        if len(time_list) == 0:
            time_list = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

        request = {
            'product_type': params['product_type'],
            'variable': era5_var,
            'year': sorted(list({str(d.year) for d in date_range})),
            'month': sorted(list({f"{d.month:02d}" for d in date_range})),
            'day': sorted(list({f"{d.day:02d}" for d in date_range})),
            'time': time_list,
            'area': area,
            'format': 'netcdf'
        }

        with st.spinner("⌛ Baixando dados do ERA5..."):
            client.retrieve('reanalysis-era5-single-levels', request, filename)

        if not os.path.exists(filename):
            st.error("❌ Arquivo não foi baixado corretamente")
            return None

        # Processar arquivo NetCDF
        with xr.open_dataset(filename) as ds:
            # Verificar e padronizar dimensão temporal
            time_dims = [dim for dim in ds.dims if 'time' in dim.lower()]
            if not time_dims:
                st.error("❌ Nenhuma dimensão temporal encontrada")
                return None
                
            if time_dims[0] != 'time':
                ds = ds.rename({time_dims[0]: 'time'})
            
            ds['time'] = pd.to_datetime(ds.time.values)
            
            # Renomear variável para nome padrão do código
            if era5_var in ds:
                ds = ds.rename({era5_var: params['precip_var']})
            
            # Converter unidades (m para mm)
            ds[params['precip_var']] = ds[params['precip_var']] * 1000
            ds[params['precip_var']].attrs['units'] = 'mm'
            
            # Salvar em um novo arquivo temporário
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False, dir=TEMP_DIR) as processed_file:
                processed_filename = processed_file.name
            
            ds.to_netcdf(processed_filename)
            
        # Remover o arquivo original temporário
        try:
            os.remove(filename)
        except:
            pass
            
        return xr.open_dataset(processed_filename)

    except Exception as e:
        st.error(f"❌ Erro no download: {str(e)}")
        logger.exception("Falha no download")
        st.warning("⚠️ Usando dados simulados para demonstração")
        return simulate_era5_data(params)

def simulate_era5_data(params):
    """Simula dados ERA5 para demonstração"""
    try:
        # Criar malha de coordenadas
        lat_min = params['lat_center'] - params['map_width']
        lat_max = params['lat_center'] + params['map_width']
        lon_min = params['lon_center'] - params['map_width']
        lon_max = params['lon_center'] + params['map_width']
        
        lats = np.linspace(lat_min, lat_max, 20)
        lons = np.linspace(lon_min, lon_max, 20)
        
        # Criar datas
        start_date = pd.to_datetime(params['start_date'])
        end_date = pd.to_datetime(params['end_date'])
        times = pd.date_range(start=start_date, end=end_date, freq='3H')
        
        # Função para gerar dados simulados de precipitação
        def generate_precip(time, lat, lon):
            # Padrão sazonal
            seasonal = np.sin(2 * np.pi * (time.dayofyear / 365)) + 1
            # Padrão espacial: mais chuva ao norte
            spatial = (lat - lat_min) / (lat_max - lat_min)
            # Aleatório
            random = np.random.exponential(1.0)
            # Alguns eventos intensos
            intense = 5 * int(np.random.rand() < 0.05)
            
            # Combinar fatores
            value = (seasonal * 0.5 + spatial * 0.3 + random * 0.2 + intense) * 2
            return max(0, value)
        
        # Gerar dados
        data = np.zeros((len(times), len(lats), len(lons)))
        
        for t, time in enumerate(times):
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    data[t, i, j] = generate_precip(time, lat, lon)
        
        # Criar dataset
        ds = xr.Dataset(
            data_vars={
                params['precip_var']: (('time', 'latitude', 'longitude'), data)
            },
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons
            }
        )
        
        # Adicionar atributos
        ds[params['precip_var']].attrs['units'] = 'mm'
        
        return ds
        
    except Exception as e:
        logger.exception(f"Erro ao simular dados: {e}")
        st.error("Erro ao gerar dados simulados")
        return None

def process_precipitation_data(ds, params):
    """Processa os dados de precipitação com verificação robusta"""
    try:
        # Verificar se a variável existe no dataset
        if params['precip_var'] not in ds.variables:
            available_vars = list(ds.variables.keys())
            st.error(f"❌ Variável não encontrada. Disponíveis: {available_vars}")
            return None

        def extract_point_data(ds, lat, lon):
            try:
                lat_idx = np.abs(ds.latitude - lat).argmin().item()
                lon_idx = np.abs(ds.longitude - lon).argmin().item()
                
                point_data = ds[params['precip_var']].isel(
                    latitude=lat_idx,
                    longitude=lon_idx
                )
                
                df = point_data.to_dataframe().reset_index()
                time_col = [col for col in df.columns if 'time' in col.lower()][0]
                
                return df.rename(columns={
                    params['precip_var']: 'precipitation',
                    time_col: 'time'
                })
                
            except Exception as e:
                logger.warning(f"Erro ao extrair dados: {str(e)}")
                return pd.DataFrame()
        
        # Processar dados
        df = extract_point_data(ds, params['lat_center'], params['lon_center'])
        if df.empty:
            return None
            
        all_regions = {}
        for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            region_df = extract_point_data(ds, lat, lon)
            if not region_df.empty:
                all_regions[region] = region_df
        
        df['date'] = df['time'].dt.date
        daily = df.groupby('date')['precipitation'].sum().reset_index()
        
        return {
            'dataset': ds,
            'timeseries': df,
            'daily': daily,
            'forecast': generate_forecast(df, params),
            'all_regions': all_regions,
            'ml_forecast': generate_ml_forecast(all_regions, params),
            'probability_map': generate_probability_map(ds, params)
        }
        
    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        logger.exception("Erro no processamento de dados")
        return None

def generate_forecast(df, params):
    """Gera previsão simples baseada em média móvel"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Criar dataframe diário completo
        df['date'] = pd.to_datetime(df['time'].dt.date)
        daily_data = df.groupby('date')['precipitation'].sum().reset_index()
        
        # Se temos menos de 7 dias de dados, não conseguimos fazer uma previsão confiável
        if len(daily_data) < 7:
            return pd.DataFrame()
        
        # Adicionar features temporais
        daily_data['dayofyear'] = daily_data['date'].dt.dayofyear
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['trend'] = np.arange(len(daily_data))
        
        # Calcular média móvel de 3 e 7 dias
        daily_data['precip_ma3'] = daily_data['precipitation'].rolling(window=3, min_periods=1).mean()
        daily_data['precip_ma7'] = daily_data['precipitation'].rolling(window=7, min_periods=1).mean()
        
        # Preparar dados para previsão
        X = daily_data[['dayofyear', 'month', 'trend', 'precip_ma3', 'precip_ma7']].values
        y = daily_data['precipitation'].values
        
        # Treinar modelo de regressão linear
        model = LinearRegression()
        model.fit(X, y)
        
        # Gerar datas futuras para previsão
        last_date = daily_data['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                    periods=params['forecast_days'])
        
        # Preparar features para previsão
        future_df = pd.DataFrame({
            'date': future_dates,
            'dayofyear': [d.dayofyear for d in future_dates],
            'month': [d.month for d in future_dates],
            'trend': np.arange(len(daily_data), len(daily_data) + len(future_dates))
        })
        
        # Inicializar com médias móveis dos últimos dias
        future_df['precip_ma3'] = daily_data['precipitation'].iloc[-3:].mean()
        future_df['precip_ma7'] = daily_data['precipitation'].iloc[-7:].mean()
        
        # Fazer previsão sequencial
        predictions = []
        
        for i in range(len(future_df)):
            X_pred = future_df.iloc[i:i+1][['dayofyear', 'month', 'trend', 'precip_ma3', 'precip_ma7']].values
            pred = max(0, model.predict(X_pred)[0])  # Não permitir precipitação negativa
            predictions.append(pred)
            
            # Atualizar médias móveis para próximas previsões
            recent_vals = list(daily_data['precipitation'].iloc[-2:].values) + predictions[:i+1]
            future_df.loc[future_df.index[i], 'precip_ma3'] = np.mean(recent_vals[-3:])
            
            recent_vals_7 = list(daily_data['precipitation'].iloc[-6:].values) + predictions[:i+1]
            future_df.loc[future_df.index[i], 'precip_ma7'] = np.mean(recent_vals_7[-7:])
        
        future_df['precipitation'] = predictions
        
        return future_df
        
    except Exception as e:
        logger.exception(f"Erro ao gerar previsão: {e}")
        return pd.DataFrame()

def generate_ml_forecast(all_regions, params):
    """Gera previsão usando machine learning para todas as regiões"""
    if not all_regions:
        return {}
    
    try:
        ml_forecasts = {}
        
        for region, df in all_regions.items():
            if df.empty:
                continue
                
            # Preparar dados diários
            df['date'] = pd.to_datetime(df['time'].dt.date)
            daily = df.groupby('date')['precipitation'].sum().reset_index()
            
            # Se temos poucos dados, pular
            if len(daily) < 7:
                continue
            
            # Criar features para o modelo
            daily['dayofyear'] = daily['date'].dt.dayofyear
            daily['month'] = daily['date'].dt.month
            daily['day'] = daily['date'].dt.day
            daily['year'] = daily['date'].dt.year
            
            # Adicionar médias móveis e lags como features
            for window in [3, 7]:
                daily[f'ma_{window}'] = daily['precipitation'].rolling(window=window, min_periods=1).mean()
            
            for lag in [1, 2, 3]:
                daily[f'lag_{lag}'] = daily['precipitation'].shift(lag).fillna(0)
            
            # Criar matriz de features e target
            features = ['dayofyear', 'month', 'day', 'ma_3', 'ma_7', 'lag_1', 'lag_2', 'lag_3']
            X = daily[features].values
            y = daily['precipitation'].values
            
            # Selecionar e treinar modelo
            if params['ml_model'] == 'RandomForest':
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
                ])
            elif params['ml_model'] == 'GradientBoosting':
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
                ])
            else:  # LinearRegression
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', LinearRegression())
                ])
            
            model.fit(X, y)
            
            # Gerar datas futuras
            last_date = daily['date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=params['forecast_days'])
            
            # Criar dataframe para previsão
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'dayofyear': [d.dayofyear for d in future_dates],
                'month': [d.month for d in future_dates],
                'day': [d.day for d in future_dates],
                'year': [d.year for d in future_dates]
            })
            
            # Inicializar com valores recentes
            forecast_df['ma_3'] = daily['precipitation'].iloc[-3:].mean()
            forecast_df['ma_7'] = daily['precipitation'].iloc[-7:].mean()
            forecast_df['lag_1'] = daily['precipitation'].iloc[-1]
            forecast_df['lag_2'] = daily['precipitation'].iloc[-2] if len(daily) > 1 else 0
            forecast_df['lag_3'] = daily['precipitation'].iloc[-3] if len(daily) > 2 else 0
            
            # Fazer previsão sequencial
            predictions = []
            
            for i in range(len(forecast_df)):
                X_pred = forecast_df.iloc[i:i+1][features].values
                pred = max(0, model.predict(X_pred)[0])  # Não permitir precipitação negativa
                predictions.append(pred)
                
                # Atualizar lags para próxima previsão
                if i + 1 < len(forecast_df):
                    forecast_df.loc[forecast_df.index[i+1], 'lag_1'] = pred
                    if i > 0:
                        forecast_df.loc[forecast_df.index[i+1], 'lag_2'] = predictions[i-1]
                    if i > 1:
                        forecast_df.loc[forecast_df.index[i+1], 'lag_3'] = predictions[i-2]
                    
                    # Atualizar médias móveis
                    if i >= 2:
                        forecast_df.loc[forecast_df.index[i+1], 'ma_3'] = np.mean(predictions[i-2:i+1])
                    if i >= 6:
                        forecast_df.loc[forecast_df.index[i+1], 'ma_7'] = np.mean(predictions[i-6:i+1])
            
            forecast_df['precipitation'] = predictions
            forecast_df['region'] = region
            
            ml_forecasts[region] = forecast_df
        
        return ml_forecasts
        
    except Exception as e:
        logger.exception(f"Erro ao gerar previsão ML: {e}")
        return {}

def generate_probability_map(ds, params):
    """Gera mapa de probabilidade de precipitação"""
    try:
        # Calcular probabilidade de chuva para cada ponto de grade
        precip_var = params['precip_var']
        threshold = 0.1  # mm - limiar para considerar como chuva
        
        # Calcular a probabilidade histórica
        prob = (ds[precip_var] > threshold).mean(dim='time') * 100
        
        # Criar figura
        fig = plt.figure(figsize=(10, 8))
        
        # Configurar mapa base
        if params['satellite_background']:
            # Usar imagem de satélite como base
            ax = plt.axes(projection=ccrs.PlateCarree())
            imagery = cimgt.GoogleTiles(style='satellite')
            ax.add_image(imagery, 8)  # Zoom level
        else:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle=':')
        
        # Configurar limites do mapa
        lat_center, lon_center = params['lat_center'], params['lon_center']
        map_width = params['map_width']
        ax.set_extent([
            lon_center - map_width, lon_center + map_width,
            lat_center - map_width, lat_center + map_width
        ])
        
        # Plotar probabilidade
        img = ax.pcolormesh(
            ds.longitude, ds.latitude, prob,
            transform=ccrs.PlateCarree(),
            cmap='YlGnBu',
            vmin=0, vmax=100,
            alpha=0.7  # Transparência para ver imagem de fundo
        )
        
        # Adicionar contorno para áreas com probabilidade acima do limiar
        ax.contour(
            ds.longitude, ds.latitude, prob, 
            levels=[params['probability_threshold']], 
            colors='red', linewidths=1
        )
        
        # Adicionar shapefile da área urbana se solicitado
        if params['show_shapefile']:
            campo_grande_gdf = get_campo_grande_shapefile()
            campo_grande_gdf.plot(
                ax=ax,
                edgecolor='black',
                facecolor='none',
                linewidth=1.5,
                transform=ccrs.PlateCarree()
            )
        
        # Adicionar colorbar
        cbar = plt.colorbar(img, ax=ax, pad=0.05)
        cbar.set_label('Probabilidade de Chuva (%)')
        
        # Adicionar pontos das regiões de interesse
        for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            ax.plot(lon, lat, 'ro', markersize=4)
            ax.text(lon + 0.01, lat + 0.01, region, fontsize=8,
                   transform=ccrs.PlateCarree(), fontweight='bold', color='white',
                   path_effects=[withStroke(linewidth=2, foreground='black')])
        
        plt.title(f"Probabilidade de Precipitação > {threshold}mm")
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.exception(f"Erro ao gerar mapa de probabilidade: {e}")
        return plt.figure()

# Função para adicionar contorno ao texto para melhor visibilidade sobre fundo de satélite
from matplotlib.patheffects import withStroke

def create_precip_map(ds, params, timestamp=None):
    """Cria mapa de precipitação para um timestamp específico ou média total"""
    try:
        # Selecionar dados para o timestamp específico ou usar média
        if timestamp is not None:
            time_index = np.abs(ds.time.values - np.datetime64(timestamp)).argmin()
            data = ds[params['precip_var']].isel(time=time_index)
            title = f"Precipitação ({params['precip_var']}) - {pd.to_datetime(timestamp).strftime('%d/%m/%Y %H:%M')}"
        else:
            data = ds[params['precip_var']].mean(dim='time')
            title = f"Precipitação Média ({params['precip_var']})"
        
        # Criar figura
        fig = plt.figure(figsize=(10, 8))
        
        # Configurar mapa base
        if params['satellite_background']:
            ax = plt.axes(projection=ccrs.PlateCarree())
            imagery = cimgt.GoogleTiles(style='satellite')
            ax.add_image(imagery, 8)
        else:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle=':')
        
        # Configurar limites do mapa
        lat_center, lon_center = params['lat_center'], params['lon_center']
        map_width = params['map_width']
        ax.set_extent([
            lon_center - map_width, lon_center + map_width,
            lat_center - map_width, lat_center + map_width
        ])
        
        # Plotar dados
        img = ax.pcolormesh(
            ds.longitude, ds.latitude, data,
            transform=ccrs.PlateCarree(),
            cmap=params['colormap'],
            vmin=0, vmax=data.max().item() * 1.1 or 10,
            alpha=0.7
        )
        
        # Adicionar shapefile da área urbana se solicitado
        if params['show_shapefile']:
            campo_grande_gdf = get_campo_grande_shapefile()
            campo_grande_gdf.plot(
                ax=ax,
                edgecolor='black',
                facecolor='none',
                linewidth=1.5,
                transform=ccrs.PlateCarree()
            )
        
        # Adicionar colorbar
        cbar = plt.colorbar(img, ax=ax, pad=0.05)
        cbar.set_label('Precipitação (mm)')
        
        # Adicionar pontos das regiões de interesse
        for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            ax.plot(lon, lat, 'ro', markersize=4)
            ax.text(lon + 0.01, lat + 0.01, region, fontsize=8,
                   transform=ccrs.PlateCarree(), fontweight='bold', color='white',
                   path_effects=[withStroke(linewidth=2, foreground='black')])
        
        plt.title(title)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.exception(f"Erro ao criar mapa: {e}")
        return plt.figure()

# Iniciar a aplicação
if __name__ == "__main__":
    main()
