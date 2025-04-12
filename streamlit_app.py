import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import io
import base64
from datetime import datetime, timedelta
from PIL import Image
import tempfile
import os
import logging
import geopandas as gpd
import cdsapi
import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from matplotlib.patheffects import withStroke

# Configuração inicial
st.set_page_config(
    layout="wide",
    page_title="Águas Guariroba - Visualizador de Precipitação - MS",
    page_icon="🌧️"
)

# Configuração básica do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Funções auxiliares para estilização
def create_gradient_background():
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
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/400x100?text=Águas+Guariroba", use_column_width=True)

# Criando a interface
def main():
    create_gradient_background()
    add_logo()
    st.title("📊 Visualizador de Precipitação - Campo Grande, MS")
    st.markdown("### Sistema de Monitoramento e Previsão de Chuvas")

    with st.sidebar:
        st.header("⚙️ Configurações")
        config_tab1, config_tab2, config_tab3 = st.tabs(["📍 Local", "📅 Período", "🔧 Avançado"])

        with config_tab1:
            area = st.selectbox(
                "Selecione a região",
                list(CAMPOS_GRANDE_AREAS.keys()),
                index=0
            )
            lat_center, lon_center = CAMPOS_GRANDE_AREAS.get(area, (-20.4697, -54.6201))
            map_width = st.slider("Área de Visualização (graus)", 0.1, 2.0, 0.3, 0.1)
            show_shapefile = st.checkbox("Mostrar Área Urbana", value=True)
            satellite_background = st.checkbox("Usar Imagem de Satélite", value=True)

        with config_tab2:
            st.subheader("Período de Análise")
            today = datetime.today()
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Início", today - timedelta(days=7))
            with col2:
                end_date = st.date_input("Fim", today + timedelta(days=5))

            st.subheader("Horários")
            all_hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
            col1, col2 = st.columns(2)
            with col1:
                start_hour = st.selectbox("Hora Inicial", all_hours)
            with col2:
                end_hour = st.selectbox("Hora Final", all_hours, index=len(all_hours)-1)

            precip_var = st.selectbox(
                "Variável de Precipitação", 
                list(PRECIPITATION_VARIABLES.keys()),
                format_func=lambda x: PRECIPITATION_VARIABLES[x]
            )
            forecast_days = st.slider("Horizonte de Previsão (dias)", 1, 14, 7)

        with config_tab3:
            st.subheader("Configurações Avançadas")
            animation_speed = st.slider("Velocidade Animação (ms)", 200, 1000, 500)
            colormap = st.selectbox("Paleta de Cores", COLORMAPS)
            product_type = st.radio("Tipo de Produto", ["reanalysis", "ensemble_mean"])
            ml_model = st.selectbox("Modelo de Previsão", 
                                  ["RandomForest", "GradientBoosting", "LinearRegression"])
            probability_threshold = st.slider("Limiar de Probabilidade (%)", 0, 100, 30)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        update_button = st.button("🔄 Atualizar Dados", use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Série Temporal", "🗺️ Mapas", "🔮 Previsões", "📊 Análise Regional"])

    if 'data' not in st.session_state or update_button:
        with st.spinner("⌛ Carregando dados..."):
            st.session_state.data = {
                'timeseries': simulate_timeseries_data(start_date, end_date),
                'daily': simulate_daily_data(start_date, end_date),
                'forecast': simulate_forecast_data(end_date, forecast_days),
                'ml_forecast': simulate_ml_forecast(list(CAMPOS_GRANDE_AREAS.keys()), end_date, forecast_days),
                'all_regions': {region: simulate_timeseries_data(start_date, end_date) 
                              for region in CAMPOS_GRANDE_AREAS.keys()}
            }

    with tab1:
        st.header(f"Série Temporal de Precipitação - {area}")
        data = st.session_state.data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(data['timeseries']['time'], data['timeseries']['precipitation'],
               width=0.02, alpha=0.7, color='#1e88e5', label='Precipitação a cada 3h')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        plt.xticks(rotation=45)
        ax.set_ylabel('Precipitação (mm)', fontsize=12)
        ax.set_title(f'Precipitação em {area} - {start_date.strftime("%d/%m/%Y")} a {end_date.strftime("%d/%m/%Y")}', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Estatísticas Diárias")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precipitação Total", f"{data['daily']['precipitation'].sum():.1f} mm")
        with col2:
            st.metric("Precipitação Média Diária", f"{data['daily']['precipitation'].mean():.1f} mm/dia")
        with col3:
            st.metric("Dias com Chuva", f"{(data['daily']['precipitation'] > 0.1).sum()} dias")

        st.subheader("Dados Diários")
        display_df = data['daily'].copy()
        display_df['date'] = display_df['date'].dt.strftime('%d/%m/%Y')
        display_df.columns = ['Data', 'Precipitação (mm)']
        st.dataframe(display_df, use_container_width=True)

    with tab2:
        st.header("Visualização Espacial da Precipitação")
        timestamps = data['timeseries']['time'].tolist()
        selected_time = st.select_slider(
            "Selecione um Momento:",
            options=timestamps,
            format_func=lambda x: x.strftime('%d/%m/%Y %H:%M')
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mapa de Precipitação")
            st.image("https://via.placeholder.com/800x600?text=Mapa+de+Precipitação", use_column_width=True)
        with col2:
            st.subheader("Probabilidade de Chuva")
            st.image("https://via.placeholder.com/800x600?text=Mapa+de+Probabilidade", use_column_width=True)

        st.subheader("Animação da Precipitação")
        animation_placeholder = st.empty()
        animation_placeholder.image("https://via.placeholder.com/800x600?text=Animação+(GIF)", use_column_width=True)
        st.download_button(
            label="⬇️ Download da Animação",
            data=io.BytesIO(b"Placeholder para o GIF real"),
            file_name="precipitacao_animacao.gif",
            mime="image/gif"
        )

    with tab3:
        st.header("Previsão de Precipitação")
        forecast_method = st.radio(
            "Método de Previsão:",
            ["Modelo Linear", "Machine Learning", "Média dos Métodos"],
            horizontal=True
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        historical_dates = data['daily']['date'].tolist()
        historical_precip = data['daily']['precipitation'].tolist()
        ax.bar(historical_dates, historical_precip, width=0.6, alpha=0.7, color='#1e88e5', label='Histórico')
        forecast_dates = data['forecast']['date'].tolist()
        forecast_precip = data['forecast']['precipitation'].tolist()
        ax.bar(forecast_dates, forecast_precip, width=0.6, alpha=0.7, color='#ff9800', label='Previsão')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        plt.xticks(rotation=45)
        ax.set_ylabel('Precipitação (mm/dia)', fontsize=12)
        ax.set_title(f'Previsão de Precipitação para {area} - Próximos {forecast_days} dias', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Detalhes da Previsão")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precipitação Total Prevista", f"{data['forecast']['precipitation'].sum():.1f} mm")
        with col2:
            st.metric("Precipitação Máxima Diária", f"{data['forecast']['precipitation'].max():.1f} mm")
        with col3:
            st.metric("Dias com Chuva Previstos", f"{(data['forecast']['precipitation'] > 0.1).sum()} dias")

        st.subheader("Dados de Previsão")
        forecast_display = data['forecast'].copy()
        forecast_display['date'] = forecast_display['date'].dt.strftime('%d/%m/%Y')
        forecast_display = forecast_display[['date', 'precipitation']]
        forecast_display.columns = ['Data', 'Precipitação Prevista (mm)']
        st.dataframe(forecast_display, use_container_width=True)

    with tab4:
        st.header("Comparação Entre Regiões")
        selected_regions = st.multiselect(
            "Selecione as regiões para comparar:",
            list(CAMPOS_GRANDE_AREAS.keys()),
            default=["Centro", "Região Norte", "Região Sul"]
        )
        if selected_regions:
            fig, ax = plt.subplots(figsize=(12, 6))
            for region in selected_regions:
                region_data = data['all_regions'][region]
                region_daily = region_data.groupby('date')['precipitation'].sum().reset_index()
                ax.plot(region_daily['date'], region_daily['precipitation'], linewidth=2, label=region)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.xticks(rotation=45)
            ax.set_ylabel('Precipitação (mm/dia)', fontsize=12)
            ax.set_title('Comparação de Precipitação Entre Regiões', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Mapa de Calor Regional")
            st.image("https://via.placeholder.com/800x600?text=Mapa+de+Calor+Regional", use_column_width=True)

            st.subheader("Tabela Comparativa")
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
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    times = pd.date_range(start=start, end=end, freq='3H')
    np.random.seed(42)
    precipitation = []
    for t in times:
        hour = t.hour
        hour_factor = 0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.2
        random_factor = np.random.exponential(1.0)
        intense = 5 * int(np.random.rand() < 0.05)
        value = max(0, hour_factor * random_factor + intense)
        precipitation.append(value)
    return pd.DataFrame({'time': times, 'precipitation': precipitation})

def simulate_daily_data(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    np.random.seed(42)
    precipitation = []
    for d in dates:
        month = d.month
        season_factor = 1.5 if (month >= 11 or month <= 3) else 0.7
        random_factor = np.random.exponential(1.0)
        value = max(0, season_factor * random_factor * 5)
        precipitation.append(value)
    return pd.DataFrame({'date': dates, 'precipitation': precipitation})

def simulate_forecast_data(end_date, forecast_days):
    last_date = pd.to_datetime(end_date)
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    np.random.seed(43)
    precipitation = []
    for d in dates:
        month = d.month
        season_factor = 1.5 if (month >= 11 or month <= 3) else 0.7
        day_factor = max(0.1, 1 - (d - dates[0]).days / forecast_days)
        random_factor = np.random.exponential(0.8)
        value = max(0, season_factor * day_factor * random_factor * 5)
        precipitation.append(value)
    return pd.DataFrame({'date': dates, 'precipitation': precipitation})

def simulate_ml_forecast(regions, end_date, forecast_days):
    ml_forecasts = {}
    for region in regions:
        last_date = pd.to_datetime(end_date)
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        np.random.seed(hash(region) % 100)
        precipitation = []
        for d in dates:
            region_factor = 1.2 if "Norte" in region else 0.8 if "Sul" in region else 1.0
            month = d.month
            season_factor = 1.5 if (month >= 11 or month <= 3) else 0.7
            random_factor = np.random.exponential(0.8)
            value = max(0, region_factor * season_factor * random_factor * 5)
            precipitation.append(value)
        ml_forecasts[region] = pd.DataFrame({
            'date': dates,
            'precipitation': precipitation,
            'region': region
        })
    return ml_forecasts

# Constantes necessárias para o funcionamento
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

CAMPO_GRANDE_BOUNDS = {
    'north': -20.35,
    'south': -20.60,
    'east': -54.50,
    'west': -54.75
}

@st.cache_data
def get_campo_grande_shapefile():
    try:
        url = "https://raw.githubusercontent.com/CampoGrandeData/GIS/main/campo_grande_urban_area.geojson"
        campo_grande_gdf = gpd.read_file(url)
        return campo_grande_gdf
    except Exception as e:
        logger.warning(f"Erro ao carregar shapefile externo: {str(e)}")
        from shapely.geometry import Polygon
        bbox = CAMPO_GRANDE_BOUNDS
        polygon = Polygon([
            (bbox['west'], bbox['north']),
            (bbox['east'], bbox['north']),
            (bbox['east'], bbox['south']),
            (bbox['west'], bbox['south'])
        ])
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        return gdf

if __name__ == "__main__":
    main()
