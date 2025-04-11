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
st.set_page_config(layout="wide", page_title="Visualizador de AOD - MS")

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
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

# --- FUNÇÕES PRINCIPAIS ---
def download_era5_data(params, client):
    """Baixa dados do ERA5 conforme parâmetros"""
    try:
        filename = f"era5_data_{params['start_date']}_{params['end_date']}.nc"
        
        # Definindo área com buffer para visualização do mapa
        buffer = params['map_width'] * 2  # Buffer para mapa mais amplo
        
        request = {
            'product_type': params['product_type'],
            'variable': params['precip_var'],
            'year': [str(d.year) for d in pd.date_range(params['start_date'], params['end_date'])],
            'month': [str(d.month) for d in pd.date_range(params['start_date'], params['end_date'])],
            'day': [str(d.day) for d in pd.date_range(params['start_date'], params['end_date'])],
            'time': [f"{h:02d}:00" for h in range(params['start_hour'], params['end_hour']+1, 3)],
            'area': [
                params['lat_center'] + buffer,
                params['lon_center'] - buffer,
                params['lat_center'] - buffer,
                params['lon_center'] + buffer
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
        
        # Extrair séries temporais para todas as regiões
        all_regions_data = {}
        for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            region_df = extract_point_data(ds, lat, lon)
            if not region_df.empty:
                all_regions_data[region] = region_df
        
        return {
            'dataset': ds,
            'timeseries': df,
            'daily': daily,
            'forecast': generate_forecast(df),
            'all_regions': all_regions_data
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
    
    predictions = np.maximum(model.predict(np.array(future_seconds).reshape(-1, 1)), 0)
    
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

def create_precipitation_map(ds, timestep, params):
    """Cria mapa de precipitação para um timestep específico"""
    fig = plt.figure(figsize=(12, 8))
    
    # Definir projeção e área
    buffer = params['map_width'] * 2
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    
    # Adicionar características do mapa
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, alpha=0.3)
    
    # Limitar área do mapa
    ax.set_extent([
        params['lon_center'] - buffer,
        params['lon_center'] + buffer,
        params['lat_center'] - buffer,
        params['lat_center'] + buffer
    ], crs=projection)
    
    # Extrair dados para o timestep
    selected_time = ds.time[timestep].values
    data = ds[params['precip_var']].isel(time=timestep) * 1000  # Convertendo para mm
    
    # Definir níveis de precipitação para o mapa de cores
    levels = [0, 0.1, 0.5, 1, 2.5, 5, 10, 15, 20, 30, 50, 75, 100]
    
    # Plotar dados
    contour = ax.contourf(
        ds.longitude, ds.latitude, data,
        levels=levels, 
        cmap=params['colormap'],
        transform=projection,
        extend='max'
    )
    
    # Adicionar marcadores para áreas de Campo Grande
    for name, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
        ax.plot(lon, lat, 'ro', markersize=4, transform=projection)
        ax.text(lon + 0.02, lat + 0.02, name, transform=projection, fontsize=8)
    
    # Adicionar barra de cores
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label(PRECIPITATION_VARIABLES[params['precip_var']])
    
    # Adicionar título com informação do timestep
    time_str = pd.to_datetime(selected_time).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Precipitação em Campo Grande - {time_str}")
    
    # Adicionar grade
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    return fig

def create_map_animation(ds, params):
    """Cria animação dos mapas de precipitação"""
    # Cria figura base
    fig = plt.figure(figsize=(12, 8))
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    
    # Buffer para área do mapa
    buffer = params['map_width'] * 2
    
    # Configurações básicas do mapa
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, alpha=0.3)
    
    ax.set_extent([
        params['lon_center'] - buffer,
        params['lon_center'] + buffer,
        params['lat_center'] - buffer,
        params['lat_center'] + buffer
    ], crs=projection)
    
    # Níveis de precipitação para o mapa de cores
    levels = [0, 0.1, 0.5, 1, 2.5, 5, 10, 15, 20, 30, 50, 75, 100]
    
    # Função para atualizar o mapa para cada frame
    def update(frame):
        ax.clear()
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, alpha=0.3)
        
        ax.set_extent([
            params['lon_center'] - buffer,
            params['lon_center'] + buffer,
            params['lat_center'] - buffer,
            params['lat_center'] + buffer
        ], crs=projection)
        
        # Extrair dados para o frame atual
        data = ds[params['precip_var']].isel(time=frame) * 1000  # Convertendo para mm
        
        # Plotar precipitação
        contour = ax.contourf(
            ds.longitude, ds.latitude, data,
            levels=levels, 
            cmap=params['colormap'],
            transform=projection,
            extend='max'
        )
        
        # Adicionar marcadores para áreas de Campo Grande
        for name, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
            ax.plot(lon, lat, 'ro', markersize=4, transform=projection)
            
        # Adicionar informação do timestamp
        time_str = pd.to_datetime(ds.time[frame].values).strftime('%Y-%m-%d %H:%M')
        ax.set_title(f"Precipitação em Campo Grande - {time_str}")
        
        # Adicionar grade
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        return contour,
    
    # Criar animação
    ani = FuncAnimation(
        fig, update,
        frames=min(10, len(ds.time)),  # Limite para os primeiros 10 frames ou menos
        interval=params['animation_speed'],
        blit=False
    )
    
    # Adicionar barra de cores
    plt.colorbar(update(0)[0], ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    
    return ani

def render_time_series(results, params):
    """Renderiza gráficos de série temporal"""
    if not results or not results['timeseries'].any():
        st.warning("Dados insuficientes para renderizar séries temporais.")
        return
    
    # Criar gráfico de série temporal principal
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Destacar região selecionada
    timeseries = results['timeseries']
    ax.plot(
        timeseries['time'], 
        timeseries['precipitation'], 
        'b-', 
        linewidth=2,
        label=params['area']
    )
    
    # Adicionar outras regiões em segundo plano para comparação (até 3)
    if 'all_regions' in results:
        other_regions = [r for r in results['all_regions'].keys() if r != params['area']][:3]
        for i, region in enumerate(other_regions):
            region_data = results['all_regions'][region]
            ax.plot(
                region_data['time'], 
                region_data['precipitation'], 
                alpha=0.5,
                linewidth=1,
                label=region
            )
    
    # Configurar formatação de data no eixo x
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
    plt.xticks(rotation=45)
    
    # Adicionar títulos e legendas
    ax.set_xlabel("Data/Hora")
    ax.set_ylabel(PRECIPITATION_VARIABLES[params['precip_var']])
    ax.set_title(f"Série Temporal de Precipitação - {params['area']}")
    ax.legend()
    
    # Adicionar grade
    ax.grid(True, alpha=0.3)
    
    # Adicionar total acumulado
    total = timeseries['precipitation'].sum()
    ax.text(
        0.02, 0.95, 
        f"Total acumulado: {total:.1f} mm",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    fig.tight_layout()
    return fig

def render_comparison_chart(results):
    """Renderiza gráfico de comparação entre regiões"""
    if not results or 'all_regions' not in results:
        return None
    
    # Calcular total acumulado por região
    region_totals = {}
    for region, data in results['all_regions'].items():
        region_totals[region] = data['precipitation'].sum()
    
    # Ordenar por total
    df = pd.DataFrame({
        'Região': region_totals.keys(),
        'Total (mm)': region_totals.values()
    }).sort_values('Total (mm)', ascending=False)
    
    # Criar gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Região'], df['Total (mm)'], color='steelblue')
    
    # Destacar região selecionada
    selected_idx = df[df['Região'] == results['timeseries'].name].index
    if not selected_idx.empty:
        bars[selected_idx[0]].set_color('firebrick')
    
    # Adicionar rótulos
    for i, v in enumerate(df['Total (mm)']):
        ax.text(i, v + 0.1, f"{v:.1f}", ha='center', fontsize=8)
    
    # Configurar eixos e títulos
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Precipitação Total (mm)")
    plt.title("Comparação da Precipitação Total por Região")
    plt.tight_layout()
    
    return fig

# --- INTERFACE DO USUÁRIO ---
def main():
    st.title("🌧️ Monitoramento de Precipitação - Campo Grande")
    st.markdown("Análise de dados de precipitação usando ERA5 do Copernicus Climate Data Store")
    
    # Inicialização - usando o cliente já inicializado no início do script
    params = setup_sidebar()
    
    # Cache para dados
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    
    # Botão de atualização principal
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Atualizar Dados", type="primary", use_container_width=True):
            with st.spinner("Baixando e processando dados..."):
                ds = download_era5_data(params, client)
                
                if ds is not None:
                    st.session_state['data'] = ds
                    st.session_state['results'] = process_precipitation_data(ds, params)
                    st.success("Dados atualizados com sucesso!")
                    st.rerun()
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise", "🗺️ Mapa", "📈 Série Temporal", "ℹ️ Sobre"])
    
    # Aba de Análise
    with tab1:
        st.header(f"Análise para {params['area']}")
        
        if st.session_state['results']:
            show_analysis_results(st.session_state['results'], params)
        else:
            st.info("Clique em 'Atualizar Dados' para carregar a análise.")
    
    # Aba de Mapa
    with tab2:
        st.header("Mapa de Precipitação")
        
        if st.session_state['data'] is not None:
            ds = st.session_state['data']
            
            # Seletor de timestamp
            timestamps = [pd.to_datetime(t.values).strftime("%Y-%m-%d %H:%M") 
                         for t in ds.time[:min(20, len(ds.time))]]
            
            selected_time = st.selectbox(
                "Selecione o horário para visualização:", 
                range(len(timestamps)), 
                format_func=lambda i: timestamps[i]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                # Toggle para animação
                show_animation = st.toggle("Mostrar animação", value=False)
            
            # Exibir mapa estático ou animação
            with st.spinner("Renderizando mapa..."):
                if show_animation:
                    try:
                        st.warning("Animação pode levar alguns instantes para ser renderizada.")
                        animation = create_map_animation(ds, params)
                        
                        # Salvar animação em buffer e exibir como vídeo
                        ani_file = f"animation_{params['start_date']}_{params['area']}.gif"
                        animation.save(ani_file, writer='pillow', fps=2)
                        
                        st.image(ani_file, caption="Animação de Precipitação", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Erro ao criar animação: {str(e)}")
                        logger.exception("Erro na animação")
                else:
                    # Mapa estático
                    fig = create_precipitation_map(ds, selected_time, params)
                    st.pyplot(fig)
        else:
            st.info("Clique em 'Atualizar Dados' para visualizar o mapa de precipitação.")
    
    # Aba de Série Temporal
    with tab3:
        st.header("Série Temporal")
        
        if st.session_state['results']:
            results = st.session_state['results']
            
            # Renderizar série temporal
            fig_ts = render_time_series(results, params)
            st.pyplot(fig_ts)
            
            # Adicionar gráfico de comparação entre regiões
            st.subheader("Comparação entre Regiões")
            fig_comp = render_comparison_chart(results)
            if fig_comp:
                st.pyplot(fig_comp)
            
            # Exibir dados tabulares
            with st.expander("Dados Detalhados"):
                st.dataframe(results['timeseries'], use_container_width=True)
        else:
            st.info("Clique em 'Atualizar Dados' para visualizar a série temporal.")
    
    # Aba Sobre
    with tab4:
        st.header("Sobre o Aplicativo")
        st.markdown("""
        ### Monitoramento de Precipitação - Campo Grande
        
        Este aplicativo utiliza dados do ERA5 do Copernicus Climate Data Store para monitorar e prever precipitação
        em diferentes regiões de Campo Grande, MS.
        
        **Funcionalidades:**
        - Visualização de dados históricos de precipitação
        - Previsão baseada em tendências recentes
        - Mapas de distribuição espacial da precipitação
        - Análise por regiões da cidade
        
        **Tecnologias utilizadas:**
        - Python
        - Streamlit
        - xarray para manipulação de dados meteorológicos
        - Cartopy para visualização geográfica
        - Scikit-learn para modelagem preditiva simples
        
        **Como usar:**
        1. Selecione os parâmetros desejados na barra lateral (região, período, variável)
        2. Clique em "Atualizar Dados" para baixar e processar os dados
        3. Navegue pelas abas para visualizar diferentes tipos de análise
        
        **Observações:**
        - Os dados do ERA5 têm resolução espacial limitada, o que pode afetar a precisão para áreas urbanas
        - A previsão é uma estimativa simples baseada em tendências recentes
        """)

if __name__ == "__main__":
    main()
