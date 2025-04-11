import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import animation
import streamlit as st
from datetime import datetime, timedelta
import os
import pandas as pd
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
import logging

# Configuração inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(layout="wide", page_title="Previsão de Precipitação - Campo Grande")

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

PRECIPITATION_VARIABLES = {
    "total_precipitation": "Precipitação Total (mm)",
    "large_scale_precipitation": "Precipitação de Grande Escala (mm)",
    "convective_precipitation": "Precipitação Convectiva (mm)"
}

COLORMAPS = ["Blues", "viridis", "plasma", "RdYlBu_r", "gist_earth"]

# --- FUNÇÕES AUXILIARES ---
def init_cds_client():
    """Inicializa o cliente CDS API com credenciais do secrets.toml"""
    try:
        if 'cds' in st.secrets:
            url = st.secrets.cds.url
            key = st.secrets.cds.key
            return cdsapi.Client(url=url, key=key)
        else:
            st.error("Credenciais do CDS não encontradas no secrets.toml")
            st.stop()
    except Exception as e:
        st.error(f"Erro ao inicializar cliente CDS: {str(e)}")
        st.stop()

def setup_sidebar():
    """Configura a barra lateral com parâmetros de entrada"""
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção de região
    area = st.sidebar.selectbox("Selecione a região", list(CAMPOS_GRANDE_AREAS.keys()))
    lat_center, lon_center = CAMPOS_GRANDE_AREAS[area]
    
    # Período de análise
    st.sidebar.subheader("Período de Análise")
    today = datetime.today()
    start_date = st.sidebar.date_input("Data Início", today)
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
    
    # Opções avançadas
    with st.sidebar.expander("Configurações Avançadas"):
        map_width = st.slider("Largura do Mapa (graus)", 0.1, 2.0, 0.3, 0.1)
        animation_speed = st.slider("Velocidade Animação (ms)", 200, 1000, 500)
        colormap = st.selectbox("Paleta de Cores", COLORMAPS)
        product_type = st.radio("Tipo de Produto", ["reanalysis", "ensemble_mean"])
    
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
        'product_type': product_type
    }

# --- FUNÇÕES PRINCIPAIS ---
def download_era5_data(params, client):
    """Baixa dados do ERA5 conforme parâmetros"""
    try:
        filename = f"era5_data_{params['start_date']}_{params['end_date']}.nc"
        
        request = {
            'product_type': params['product_type'],
            'variable': params['precip_var'],
            'year': [str(d.year) for d in pd.date_range(params['start_date'], params['end_date'])],
            'month': [str(d.month) for d in pd.date_range(params['start_date'], params['end_date'])],
            'day': [str(d.day) for d in pd.date_range(params['start_date'], params['end_date'])],
            'time': [f"{h:02d}:00" for h in range(params['start_hour'], params['end_hour']+1, 3)],
            'area': [
                params['lat_center'] + params['map_width']/2,
                params['lon_center'] - params['map_width']/2,
                params['lat_center'] - params['map_width']/2,
                params['lon_center'] + params['map_width']/2
            ],
            'format': 'netcdf'
        }
        
        with st.spinner("Baixando dados do ERA5..."):
            client.retrieve("reanalysis-era5-single-levels", request, filename)
        
        return xr.open_dataset(filename)
    
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        logger.exception("Erro no download de dados")
        return None

def process_precipitation_data(ds, params):
    """Processa os dados de precipitação"""
    try:
        # Extrai série temporal para o ponto central
        def extract_point_data(ds, lat, lon):
            lat_idx = np.abs(ds.latitude - lat).argmin().item()
            lon_idx = np.abs(ds.longitude - lon).argmin().item()
            time_dim = next((dim for dim in ['time', 'forecast_time'] if dim in ds.dims), None)
            
            if not time_dim:
                return pd.DataFrame()
            
            data = ds[params['precip_var']].isel(
                latitude=lat_idx, 
                longitude=lon_idx
            ).to_dataframe().reset_index()
            
            # Converter m para mm se necessário
            if params['precip_var'] in PRECIPITATION_VARIABLES:
                data[params['precip_var']] *= 1000
            
            return data.rename(columns={params['precip_var']: 'precipitation', time_dim: 'time'})
        
        df = extract_point_data(ds, params['lat_center'], params['lon_center'])
        
        if df.empty:
            return None
            
        # Calcula estatísticas diárias
        df['date'] = df['time'].dt.date
        daily = df.groupby('date')['precipitation'].sum().reset_index()
        
        return {
            'dataset': ds,
            'timeseries': df,
            'daily': daily,
            'forecast': generate_forecast(df)
        }
    
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        logger.exception("Erro no processamento")
        return None

def generate_forecast(df, days=3):
    """Gera previsão simples baseada em tendências"""
    if len(df) < 3:
        return pd.DataFrame()
    
    # Modelo de regressão
    df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()
    model = LinearRegression().fit(df[['time_numeric']], df['precipitation'])
    
    # Gerar previsão
    last_time = df['time'].max()
    future_times = [last_time + timedelta(hours=6*i) for i in range(1, days*4+1)]
    future_seconds = [(t - df['time'].min()).total_seconds() for t in future_times]
    
    predictions = np.maximum(model.predict(np.array(future_seconds).reshape(-1, 1), 0))
    
    # Adicionar sazonalidade diária
    for i, t in enumerate(future_times):
        hour_factor = 1 + 0.5 * np.sin(np.pi * (t.hour - 6) / 12)
        predictions[i] *= hour_factor
    
    # Combinar dados
    forecast = pd.DataFrame({
        'time': future_times,
        'precipitation': predictions.flatten(),
        'type': 'forecast'
    })
    
    return pd.concat([
        df[['time', 'precipitation']].assign(type='historical'),
        forecast
    ], ignore_index=True)

# --- INTERFACE DO USUÁRIO ---
def main():
    st.title("🌧️ Monitoramento de Precipitação - Campo Grande")
    st.markdown("Análise de dados de precipitação usando ERA5 do Copernicus Climate Data Store")
    
    # Inicialização
    client = init_cds_client()
    params = setup_sidebar()
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise", "🗺️ Mapa", "📈 Série Temporal", "ℹ️ Sobre"])
    
    with tab1:
        st.header(f"Análise para {params['area']}")
        
        if st.button("🔄 Atualizar Análise"):
            with st.spinner("Processando..."):
                ds = download_era5_data(params, client)
                
                if ds is not None:
                    results = process_precipitation_data(ds, params)
                    
                    if results:
                        show_analysis_results(results, params)
                    else:
                        st.error("Dados insuficientes para análise")

def show_analysis_results(results, params):
    """Exibe os resultados da análise"""
    # Estatísticas resumidas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Precipitação Diária")
        if not results['daily'].empty:
            daily = results['daily'].copy()
            daily['emoji'] = daily['precipitation'].apply(
                lambda x: "☀️" if x == 0 else 
                "🌦️" if x < 2.5 else 
                "🌧️" if x < 10 else 
                "⛈️" if x < 25 else "🌊"
            )
            st.dataframe(daily, use_container_width=True)
    
    with col2:
        st.subheader("Estatísticas")
        if not results['timeseries'].empty:
            total = results['timeseries']['precipitation'].sum()
            st.metric("Precipitação Total", f"{total:.1f} mm")
            st.metric("Máxima em 3h", f"{results['timeseries']['precipitation'].max():.1f} mm")
    
    # Previsão
    st.subheader("Previsão")
    if not results['forecast'].empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        hist = results['forecast'][results['forecast']['type'] == 'historical']
        pred = results['forecast'][results['forecast']['type'] == 'forecast']
        
        ax.plot(hist['time'], hist['precipitation'], 'b-', label='Histórico')
        ax.plot(pred['time'], pred['precipitation'], 'r--', label='Previsão')
        
        ax.set_xlabel("Data")
        ax.set_ylabel("Precipitação (mm)")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
