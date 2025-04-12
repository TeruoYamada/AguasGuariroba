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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.colors import LinearSegmentedColormap
import logging
import io
import tempfile
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import geopandas as gpd
from PIL import Image
import base64
from io import BytesIO
from scipy import stats

# Configuração inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(layout="wide", page_title="Águas Guariroba - Visualizador de Precipitação - MS")

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
        'probability_threshold': probability_threshold
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
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Configurar limites do mapa
        lat_center, lon_center = params['lat_center'], params['lon_center']
        map_width = params['map_width']
        ax.set_extent([
            lon_center - map_width, lon_center + map_width,
            lat_center - map_width, lat_center + map_width
        ])
        
        # Adicionar características do mapa
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':')
        
        # Plotar probabilidade
        img = ax.pcolormesh(
            ds.longitude, ds.latitude, prob,
            transform=ccrs.PlateCarree(),
            cmap='YlGnBu',
            vmin=0, vmax=100
        )
        
        # Adicionar contorno para áreas com probabilidade acima do limiar
        ax.contour(
            ds.longitude, ds.latitude, prob, 
            levels=[params['probability_threshold']], 
            colors='red', linewidths=1
        )
        
        # Adicionar colorbar
        cbar = plt.colorbar(img, ax=ax, pad=0.05)
        cbar.set_label('Probabilidade de Chuva (%)')
        
        # Adicionar pontos das regiões de interesse
        for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            ax.plot(lon, lat, 'ro', markersize=4)
            ax.text(lon + 0.01, lat + 0.01, region, fontsize=8,
                   transform=ccrs.PlateCarree())
        
        plt.title(f"Probabilidade de Precipitação > {threshold}mm")
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.exception(f"Erro ao gerar mapa de probabilidade: {e}")
        return plt.figure()

def create_precipitation_map(ds, time_idx, params):
    """Cria mapa de precipitação para um horário específico"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Adicionar características do mapa
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
        
        # Plotar dados de precipitação
        precip = ds[params['precip_var']].isel(time=time_idx)
        max_val = np.percentile(precip.values, 95) if not np.all(precip.values == 0) else 1
        
        img = ax.pcolormesh(
            ds.longitude, ds.latitude, precip,
            transform=ccrs.PlateCarree(),
            cmap=params['colormap'],
            vmin=0, vmax=max_val
        )
        
        # Adicionar colorbar
        cbar = plt.colorbar(img, ax=ax, pad=0.05)
        cbar.set_label(PRECIPITATION_VARIABLES[params['precip_var']])
        
        # Adicionar título
        time_str = pd.to_datetime(ds.time[time_idx].values).strftime('%Y-%m-%d %H:%M')
        plt.title(f"{PRECIPITATION_VARIABLES[params['precip_var']]} - {time_str}")
        
        # Adicionar pontos das regiões de interesse
        for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            ax.plot(lon, lat, 'ro', markersize=4)
            ax.text(lon + 0.01, lat + 0.01, region, fontsize=8,
                   transform=ccrs.PlateCarree())
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.exception(f"Erro ao criar mapa: {e}")
        st.error(f"Erro ao criar mapa: {str(e)}")
        return plt.figure()

def create_map_animation(ds, params):
    """Cria animação do mapa de precipitação"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Configurar limites do mapa
        lat_center, lon_center = params['lat_center'], params['lon_center']
        map_width = params['map_width']
        ax.set_extent([
            lon_center - map_width, lon_center + map_width,
            lat_center - map_width, lat_center + map_width
        ])
        
        # Adicionar características do mapa
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':')
        
        # Determinar valores max/min para colorbar consistente
        max_val = np.percentile(ds[params['precip_var']].values, 95)
        if np.isnan(max_val) or max_val == 0:
            max_val = 1
        
        # Função de atualização para animação
        def update(frame):
            ax.clear()
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle=':')
            ax.set_extent([
                lon_center - map_width, lon_center + map_width,
                lat_center - map_width, lat_center + map_width
            ])
            
            # Plotar dados
            precip = ds[params['precip_var']].isel(time=frame)
            img = ax.pcolormesh(
                ds.longitude, ds.latitude, precip,
                transform=ccrs.PlateCarree(),
                cmap=params['colormap'],
                vmin=0, vmax=max_val
            )
            
            # Adicionar título com timestamp
            time_str = pd.to_datetime(ds.time[frame].values).strftime('%Y-%m-%d %H:%M')
            ax.set_title(f"{PRECIPITATION_VARIABLES[params['precip_var']]} - {time_str}")
            
            # Adicionar pontos das regiões
            for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
                ax.plot(lon, lat, 'ro', markersize=4)
                ax.text(lon + 0.01, lat + 0.01, region, fontsize=8,
                      transform=ccrs.PlateCarree())
            
            return [img]
        
        # Criar animação
        frames = min(20, len(ds.time))  # Limitar a 20 frames para performance
        ani = FuncAnimation(
            fig, update, frames=frames, 
            blit=False, interval=params['animation_speed']
        )
        
        plt.close()  # Evitar exibição duplicada
        return ani
        
    except Exception as e:
        logger.exception(f"Erro na animação: {e}")
        st.error(f"Erro ao criar animação: {str(e)}")
        return None

def render_time_series(results, params):
    """Renderiza gráfico de série temporal"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotar dados históricos
    df = results['timeseries']
    ax.bar(df['time'], df['precipitation'], width=0.02, color='blue', alpha=0.7, label='Observado')
    
    # Plotar previsão se disponível
    if not results['forecast'].empty:
        forecast = results['forecast']
        ax.bar(forecast['date'], forecast['precipitation'], width=0.8, 
               color='orange', alpha=0.5, label='Previsão')
    
    # Configurar eixos e título
    ax.set_xlabel('Data e Hora')
    ax.set_ylabel('Precipitação (mm)')
    ax.set_title(f"Série Temporal de Precipitação - {params['area']}")
    ax.legend()
    
    # Formatar eixo x para datas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    
    return fig

def render_comparison_chart(results):
    """Renderiza gráfico de comparação entre regiões"""
    if not results.get('all_regions'):
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for region, df in results['all_regions'].items():
        daily = df.groupby(df['time'].dt.date)['precipitation'].sum()
        ax.plot(daily.index, daily.values, label=region)
    
    ax.set_xlabel('Data')
    ax.set_ylabel('Precipitação Acumulada (mm)')
    ax.set_title('Comparação de Precipitação entre Regiões')
    ax.legend()
    fig.autofmt_xdate()
    
    return fig

def show_analysis_results(results, params):
    """Mostra os resultados da análise"""
    if not results:
        st.warning("Nenhum resultado disponível para exibição")
        return
    
    # Resumo estatístico
    st.subheader("📊 Resumo Estatístico")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_precip = results['daily']['precipitation'].sum()
        st.metric("Precipitação Total", f"{total_precip:.1f} mm")
    
    with col2:
        max_daily = results['daily']['precipitation'].max()
        st.metric("Máximo Diário", f"{max_daily:.1f} mm")
    
    with col3:
        rain_days = (results['daily']['precipitation'] > 0.1).sum()
        st.metric("Dias com Chuva", f"{rain_days} dias")
    
    # Previsões
    st.subheader("🔮 Previsões")
    if not results['forecast'].empty:
        forecast = results['forecast']
        
        cols = st.columns(len(forecast))
        for i, (_, row) in enumerate(forecast.iterrows()):
            with cols[i]:
                date_str = row['date'].strftime('%d/%m')
                st.metric(
                    date_str, 
                    f"{row['precipitation']:.1f} mm", 
                    delta=None
                )
        
        # Gráfico de previsão
        st.subheader("📈 Previsão para os Próximos Dias")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(forecast['date'], forecast['precipitation'], color='orange', alpha=0.7)
        ax.set_title(f"Previsão de Precipitação - {params['area']}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Precipitação (mm)")
        fig.autofmt_xdate()
        st.pyplot(fig)
    else:
        st.info("Dados históricos insuficientes para gerar previsão")
    
    # ML Forecast por região
    st.subheader("🧠 Previsão por Modelo de Machine Learning")
    if results['ml_forecast']:
        # Criar um gráfico comparativo das previsões
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for region, forecast_df in results['ml_forecast'].items():
            ax.plot(forecast_df['date'], forecast_df['precipitation'], 
                   marker='o', label=region)
        
        ax.set_xlabel("Data")
        ax.set_ylabel("Precipitação Prevista (mm)")
        ax.set_title(f"Previsão por {params['ml_model']} - Todas as Regiões")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)
        
        # Tabela de dados
        st.subheader("📋 Tabela de Previsões")
        forecast_summary = pd.DataFrame()
        
        for region, df in results['ml_forecast'].items():
            if forecast_summary.empty:
                forecast_summary = pd.DataFrame({'date': df['date']})
            forecast_summary[region] = df['precipitation'].round(1)
        
        forecast_summary = forecast_summary.set_index('date')
        st.dataframe(forecast_summary)
    else:
        st.info("Dados insuficientes para gerar previsão por ML")
    
    # Mapa de probabilidade
    st.subheader("🌧️ Mapa de Probabilidade de Precipitação")
    if 'probability_map' in results:
        st.pyplot(results['probability_map'])
    
    # Série temporal
    st.subheader("⏱️ Série Temporal")
    st.pyplot(render_time_series(results, params))
    
    # Comparação entre regiões
    st.subheader("🔄 Comparação entre Regiões")
    comparison_chart = render_comparison_chart(results)
    if comparison_chart:
        st.pyplot(comparison_chart)
    
    # Mapa de precipitação
    st.subheader("🗺️ Mapa de Precipitação")
    
    # Seletor de tempo
    times = pd.to_datetime(results['dataset'].time.values)
    time_options = [t.strftime('%Y-%m-%d %H:%M') for t in times]
    selected_time = st.selectbox("Selecione o horário", time_options)
    selected_idx = time_options.index(selected_time)
    
    # Mostrar mapa para o horário selecionado
    st.pyplot(create_precipitation_map(results['dataset'], selected_idx, params))
    
    # Animação
    st.subheader("🎬 Animação")
    ani = create_map_animation(results['dataset'], params)
    if ani:
        # Salvar animação como gif e exibir
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
            ani_filename = temp_file.name
        
        ani.save(ani_filename, writer='pillow', fps=2)
        
        # Exibir animação
        file_ = open(ani_filename, "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="animação de precipitação">',
            unsafe_allow_html=True,
        )
        
        try:
            os.remove(ani_filename)
        except:
            pass
    else:
        st.warning("Não foi possível criar a animação")

def show_region_analysis(results, params):
    """Mostra análise detalhada por região"""
    if not results or not results.get('all_regions'):
        st.warning("Dados por região não disponíveis")
        return
    
    # Selecionar região
    regions = list(results['all_regions'].keys())
    selected_region = st.selectbox("Selecione a região para análise detalhada", regions)
    
    if selected_region not in results['all_regions']:
        st.warning(f"Dados para {selected_region} não disponíveis")
        return
    
    # Obter dados da região selecionada
    region_df = results['all_regions'][selected_region]
    region_daily = region_df.groupby(region_df['time'].dt.date)['precipitation'].sum().reset_index()
    region_daily.columns = ['date', 'precipitation']
    
    # Estatísticas
    st.subheader(f"📊 Estatísticas para {selected_region}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", f"{region_daily['precipitation'].sum():.1f} mm")
    
    with col2:
        st.metric("Máximo Diário", f"{region_daily['precipitation'].max():.1f} mm")
    
    with col3:
        st.metric("Média Diária", f"{region_daily['precipitation'].mean():.1f} mm")
    
    with col4:
        rain_days = (region_daily['precipitation'] > 0.1).sum()
        st.metric("Dias com Chuva", f"{rain_days} dias")
    
    # Gráfico de série temporal
    st.subheader("📈 Série Temporal")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(region_daily['date'], region_daily['precipitation'], alpha=0.7)
    ax.set_xlabel("Data")
    ax.set_ylabel("Precipitação (mm)")
    ax.set_title(f"Precipitação Diária - {selected_region}")
    fig.autofmt_xdate()
    st.pyplot(fig)
    
    # Previsão ML
    if results['ml_forecast'] and selected_region in results['ml_forecast']:
        st.subheader("🔮 Previsão ML")
        ml_forecast = results['ml_forecast'][selected_region]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(ml_forecast['date'], ml_forecast['precipitation'], color='orange', alpha=0.7)
        ax.set_xlabel("Data")
        ax.set_ylabel("Precipitação Prevista (mm)")
        ax.set_title(f"Previsão {params['ml_model']} - {selected_region}")
        fig.autofmt_xdate()
        st.pyplot(fig)
        
        # Tabela de previsão
        st.subheader("📋 Tabela de Previsão")
        forecast_table = ml_forecast[['date', 'precipitation']].copy()
        forecast_table['precipitation'] = forecast_table['precipitation'].round(1)
        forecast_table.columns = ['Data', 'Precipitação (mm)']
        st.dataframe(forecast_table)

def display_about():
    """Exibe informações sobre o aplicativo"""
    st.subheader("Sobre este Aplicativo")
    
    st.markdown("""
    ### 📊 Visualizador de Precipitação - Campo Grande, MS
    
    Este aplicativo permite visualizar dados históricos e previsões de precipitação para 
    Campo Grande e regiões, utilizando dados do ERA5 da Copernicus Climate Data Store.
    
    #### Funcionalidades:
    
    - 🗺️ **Mapas de Precipitação**: Visualização espacial da precipitação
    - 📈 **Séries Temporais**: Análise temporal da precipitação
    - 🔮 **Previsão**: Modelos de ML para previsão de precipitação
    - 🌧️ **Probabilidade**: Mapas de probabilidade de chuva
    - 📊 **Análise Regional**: Comparação entre diferentes regiões
    
    #### Dados:
    
    Os dados são obtidos do ERA5, o modelo mais recente de reanálise atmosférica 
    produzido pelo Centro Europeu de Previsões Meteorológicas de Médio Prazo (ECMWF).
    
    #### Como usar:
    
    1. Selecione a região de interesse no painel lateral
    2. Escolha o período de análise
    3. Configure os parâmetros avançados se necessário
    4. Explore os diferentes painéis de visualização
    
    ---
    
    Desenvolvido para Águas Guariroba S.A.
    """)

def main():
    """Função principal"""
    # Logo e título
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("https://aguasguariroba.com.br/wp-content/uploads/2019/08/logo-menu.png", width=100)
    
    with col2:
        st.title("Visualizador de Precipitação - Campo Grande")
        st.caption("Análise e previsão de precipitação para o sistema de abastecimento")
    
    # Abas
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Análise Regional", "ℹ️ Sobre"])
    
    # Configuração da barra lateral
    params = setup_sidebar()
    
    # Obter cliente CDS
    cds_client = get_cds_client()
    
    with tab1:
        # Download e processamento de dados
        with st.spinner("Carregando dados..."):
            ds = download_era5_data(params, cds_client)
            
            if ds is not None:
                results = process_precipitation_data(ds, params)
                if results:
                    show_analysis_results(results, params)
                else:
                    st.error("❌ Erro no processamento dos dados")
            else:
                st.error("❌ Não foi possível obter os dados")
    
    with tab2:
        # Análise regional
        if 'results' in locals() and results:
            show_region_analysis(results, params)
        else:
            with st.spinner("Carregando dados..."):
                ds = download_era5_data(params, cds_client)
                
                if ds is not None:
                    results = process_precipitation_data(ds, params)
                    if results:
                        show_region_analysis(results, params)
                    else:
                        st.error("❌ Erro no processamento dos dados")
                else:
                    st.error("❌ Não foi possível obter os dados")
    
    with tab3:
        display_about()

if __name__ == "__main__":
    main()
        
