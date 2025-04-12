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
import io
from matplotlib.animation import FuncAnimation

# Configuração inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(layout="wide", page_title="Águas Guariroba - Visualizador de Precipitação - MS")

# ✅ Autenticação CDS (ERA5)
try:
    client_cds = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=st.secrets["cds"]["key"]
    )
except Exception as e:
    st.error(f"❌ Erro ao conectar ao Climate Data Store: {str(e)}")
    st.stop()

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

def download_era5_data(params, client):
    """Baixa dados do ERA5 com tratamento robusto de variáveis"""
    try:
        # Verificar se as datas são válidas
        if params['start_date'] > params['end_date']:
            st.error("❌ Data de início maior que data de fim")
            return None

        # Obter nome correto da variável no ERA5
        era5_var = ERA5_VARIABLES.get(params['precip_var'])
        if not era5_var:
            st.error(f"❌ Variável {params['precip_var']} não mapeada")
            return None

        filename = f"era5_data_{params['start_date']}_{params['end_date']}.nc"
        area = [
            params['lat_center'] + 0.15,
            params['lon_center'] - 0.15,
            params['lat_center'] - 0.15,
            params['lon_center'] + 0.15
        ]

        date_range = pd.date_range(params['start_date'], params['end_date'])
        time_list = [f"{h:02d}:00" for h in range(params['start_hour'], params['end_hour'] + 1, 3)]

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
            ds = ds.rename({era5_var: params['precip_var']})
            
            # Converter unidades (m para mm)
            ds[params['precip_var']] = ds[params['precip_var']] * 1000
            ds[params['precip_var']].attrs['units'] = 'mm'
            
            ds.to_netcdf(filename)
            
        return xr.open_dataset(filename)

    except Exception as e:
        st.error(f"❌ Erro no download: {str(e)}")
        logger.exception("Falha no download")
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
            'forecast': generate_forecast(df),
            'all_regions': all_regions
        }
        
    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        return None

# ... (mantenha as outras funções como create_precipitation_map, create_map_animation, 
# render_time_series, render_comparison_chart e show_analysis_results exatamente como estão)

def main():
    st.title("🌧️ Monitoramento de Precipitação - Campo Grande")
    st.markdown("Análise de dados de precipitação usando ERA5 do Copernicus Climate Data Store")
    
    params = setup_sidebar()
    
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Atualizar Dados", type="primary", use_container_width=True):
            with st.spinner("Baixando e processando dados..."):
                ds = download_era5_data(params, client_cds)
                
                if ds is not None:
                    st.session_state['data'] = ds
                    st.session_state['results'] = process_precipitation_data(ds, params)
                    st.success("Dados atualizados com sucesso!")
                    st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise", "🗺️ Mapa", "📈 Série Temporal", "ℹ️ Sobre"])
    
    with tab1:
        st.header(f"Análise para {params['area']}")
        if st.session_state.get('results'):
            show_analysis_results(st.session_state['results'], params)
        else:
            st.info("Clique em 'Atualizar Dados' para carregar a análise.")
    
    with tab2:
        st.header("Mapa de Precipitação")
        if st.session_state.get('data') is not None:
            ds = st.session_state['data']
            
            try:
                if hasattr(ds, 'time') and len(ds.time) > 0:
                    timestamps = [pd.to_datetime(t.values).strftime("%Y-%m-%d %H:%M") 
                                for t in ds.time[:min(20, len(ds.time))]]
                    
                    selected_time = st.selectbox(
                        "Selecione o horário:", 
                        range(len(timestamps)), 
                        format_func=lambda i: timestamps[i]
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        show_animation = st.toggle("Mostrar animação", value=False)
                    
                    with st.spinner("Renderizando mapa..."):
                        if show_animation:
                            try:
                                animation = create_map_animation(ds, params)
                                ani_file = f"animation_{params['start_date']}_{params['area']}.gif"
                                animation.save(ani_file, writer='pillow', fps=2)
                                st.image(ani_file, use_column_width=True)
                            except Exception as e:
                                st.error(f"Erro na animação: {str(e)}")
                        else:
                            fig = create_precipitation_map(ds, selected_time, params)
                            st.pyplot(fig)
                else:
                    st.error("Dados temporais inválidos")
            except Exception as e:
                st.error(f"Erro: {str(e)}")
        else:
            st.info("Clique em 'Atualizar Dados' para visualizar o mapa.")
    
    with tab3:
        st.header("Série Temporal")
        if st.session_state.get('results'):
            results = st.session_state['results']
            fig_ts = render_time_series(results, params)
            st.pyplot(fig_ts)
            
            st.subheader("Comparação entre Regiões")
            fig_comp = render_comparison_chart(results)
            if fig_comp:
                st.pyplot(fig_comp)
            
            with st.expander("Dados Detalhados"):
                st.dataframe(results['timeseries'], use_container_width=True)
        else:
            st.info("Clique em 'Atualizar Dados' para visualizar a série temporal.")
    
    with tab4:
        st.header("Sobre o Aplicativo")
        st.markdown("""
        ### Monitoramento de Precipitação - Campo Grande
        Aplicativo utilizando dados ERA5 do Copernicus Climate Data Store.
        """)

if __name__ == "__main__":
    main()
