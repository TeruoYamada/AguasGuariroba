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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração inicial
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

# --- FUNÇÕES DE SIMULAÇÃO PARA DEMONSTRAÇÃO ---
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

def generate_era5_data(params):
    """Gera dados simulados para substituir os dados do ERA5 quando não disponíveis"""
    try:
        st.warning("⚠️ Usando dados simulados para demonstração")
        
        # Simula série temporal para a área selecionada
        timeseries_data = simulate_timeseries_data(params['start_date'], params['end_date'])
        
        # Simula dados diários
        daily_data = simulate_daily_data(params['start_date'], params['end_date'])
        
        # Simula dados de previsão
        forecast_data = simulate_forecast_data(params['end_date'], params['forecast_days'])
        
        # Simula dados para todas as regiões
        all_regions_data = {}
        for region in CAMPOS_GRANDE_AREAS.keys():
            all_regions_data[region] = simulate_timeseries_data(params['start_date'], params['end_date'])
        
        # Simula previsões ML
        ml_forecast_data = simulate_ml_forecast(
            list(CAMPOS_GRANDE_AREAS.keys()), 
            params['end_date'],
            params['forecast_days']
        )
        
        return {
            'timeseries': timeseries_data,
            'daily': daily_data,
            'forecast': forecast_data,
            'all_regions': all_regions_data,
            'ml_forecast': ml_forecast_data
        }
        
    except Exception as e:
        logger.exception(f"Erro ao gerar dados simulados: {e}")
        st.error(f"Erro ao gerar dados simulados: {str(e)}")
        return None

def try_import_optional_modules():
    """Tenta importar módulos opcionais e retorna um dicionário com o status"""
    modules = {}
    
    try:
        import cdsapi
        modules['cdsapi'] = True
    except ImportError:
        modules['cdsapi'] = False
        
    try:
        import xarray as xr
        modules['xarray'] = True
    except ImportError:
        modules['xarray'] = False
        
    try:
        import geopandas as gpd
        import matplotlib.patheffects
        modules['geopandas'] = True
    except ImportError:
        modules['geopandas'] = False
        
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.img_tiles as cimgt
        modules['cartopy'] = True
    except ImportError:
        modules['cartopy'] = False
        
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        modules['sklearn'] = True
    except ImportError:
        modules['sklearn'] = False
        
    return modules

# Função para criar um mapa simplificado quando o Cartopy não está disponível
def create_simple_map(data, title):
    """Cria um mapa simplificado sem o Cartopy"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Criar um gráfico de calor simples
    x = np.linspace(-54.75, -54.50, 20)
    y = np.linspace(-20.60, -20.35, 20)
    X, Y = np.meshgrid(x, y)
    
    # Gerar dados aleatórios
    np.random.seed(42)
    Z = np.random.rand(20, 20) * 10
    
    # Criar o mapa de calor
    c = ax.pcolormesh(X, Y, Z, cmap='Blues', alpha=0.7)
    
    # Adicionar pontos representando as regiões
    for region, (lat, lon) in CAMPOS_GRANDE_AREAS.items():
        ax.plot(lon, lat, 'ro', markersize=4)
        ax.text(lon + 0.01, lat + 0.01, region, fontsize=8, 
               fontweight='bold', color='black')
    
    # Formatar o gráfico
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    # Adicionar colorbar
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label('Precipitação (mm)')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

# Função principal da aplicação
def main():
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
    if not modules['xarray']:
        st.warning("⚠️ O módulo xarray não está instalado. Alguns recursos podem não funcionar.")
    
    if not modules['cdsapi']:
        st.warning("⚠️ O módulo cdsapi não está instalado. Usando dados simulados.")
    
    if not modules['cartopy']:
        st.warning("⚠️ O módulo cartopy não está instalado. Mapas simplificados serão usados.")
    
    if not modules['geopandas']:
        st.warning("⚠️ O módulo geopandas não está instalado. Shapefiles não serão utilizados.")

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
            st.subheader("Dados de Previsão")
            forecast_display = data['forecast'].copy()
            
            # Converter datas para string no formato brasileiro
            if 'date' in forecast_display.columns:
                if pd.api.types.is_datetime64_any_dtype(forecast_display['date']):
                    forecast_display['date'] = forecast_display['date'].dt.strftime('%d/%m/%Y')
            
            forecast_display = forecast_display[['date', 'precipitation']]
            forecast_display.columns = ['Data', 'Precipitação Prevista (mm)']
            st.dataframe(forecast_display, use_container_width=True)
        else:
            st.warning("Dados de previsão não estão disponíveis.")
    
    # Tab 4: Análise Regional
    with tab4:
        st.header("Comparação Entre Regiões")
        
        # Selecionar regiões para comparação
        selected_regions = st.multiselect(
            "Selecione as regiões para comparar:",
            list(CAMPOS_GRANDE_AREAS.keys()),
            default=["Centro", "Região Norte", "Região Sul"]
        )
        
        if data and 'all_regions' in data and selected_regions:
            # Verificar se todas as regiões selecionadas existem nos dados
            valid_regions = [r for r in selected_regions if r in data['all_regions'] and not data['all_regions'][r].empty]
            
            if valid_regions:
                # Gráfico de comparação entre regiões
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for region in valid_regions:
                    region_data = data['all_regions'][region]
                    # Agrupar por data se os dados forem por hora
                    if 'time' in region_data.columns:
                        region_data['date'] = region_data['time'].dt.date
                        region_daily = region_data.groupby('date')['precipitation'].sum().reset_index()
                        region_daily['date'] = pd.to_datetime(region_daily['date'])
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
                
                if modules['cartopy']:
                    st.image("https://via.placeholder.com/800x600?text=Mapa+de+Calor+Regional", use_column_width=True)
                else:
                    # Usar mapa simplificado
                    fig = create_simple_map(data, "Mapa de Calor Regional")
                    st.pyplot(fig)
                
                st.subheader("Estatísticas por Região")
                
                # Preparar dados para tabela comparativa
                regions_stats = []
                for region in valid_regions:
                    region_data = data['all_regions'][region]
                    if 'time' in region_data.columns:
                        region_data['date'] = region_data['time'].dt.date
                        region_daily = region_data.groupby('date')['precipitation'].sum().reset_index()
                    else:
                        region_daily = region_data
                    
                    # Calcular estatísticas
                    total = region_daily['precipitation'].sum()
                    avg = region_daily['precipitation'].mean()
                    max_val = region_daily['precipitation'].max()
                    rainy_days = (region_daily['precipitation'] > 0.1).sum()
                    
                    regions_stats.append({
                        'Região': region,
                        'Precipitação Total (mm)': round(total, 1),
                        'Média Diária (mm)': round(avg, 1),
                        'Máxima Diária (mm)': round(max_val, 1),
                        'Dias com Chuva': rainy_days
                    })
                
                # Criar DataFrame e exibir
                stats_df = pd.DataFrame(regions_stats)
                st.dataframe(stats_df, use_container_width=True)
                
                # Adicionar previsões regionais se disponíveis
                if 'ml_forecast' in data and data['ml_forecast']:
                    st.subheader("Previsão Regional")
                    
                    # Filtrar previsões para regiões selecionadas
                    selected_forecasts = {r: data['ml_forecast'][r] for r in valid_regions if r in data['ml_forecast']}
                    
                    if selected_forecasts:
                        # Gráfico de previsão por região
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for region, forecast_data in selected_forecasts.items():
                            ax.plot(forecast_data['date'], forecast_data['precipitation'], 
                                   linewidth=2, label=f"{region} (Previsão)")
                        
                        # Formatar eixos
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
                        plt.xticks(rotation=45)
                        ax.set_ylabel('Precipitação Prevista (mm/dia)', fontsize=12)
                        ax.set_title('Previsão de Precipitação por Região', fontsize=14)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Tabela de previsão por região
                        st.subheader("Dados de Previsão por Região")
                        
                        # Combinar todas as previsões em uma tabela
                        all_forecasts = []
                        for region, forecast_data in selected_forecasts.items():
                            region_forecast = forecast_data.copy()
                            region_forecast['region'] = region
                            all_forecasts.append(region_forecast)
                        
                        if all_forecasts:
                            combined_forecast = pd.concat(all_forecasts)
                            pivot_forecast = combined_forecast.pivot(index='date', columns='region', values='precipitation')
                            
                            # Formatar datas
                            pivot_forecast.index = pivot_forecast.index.strftime('%d/%m/%Y')
                            
                            # Arredondar valores
                            pivot_forecast = pivot_forecast.round(1)
                            
                            st.dataframe(pivot_forecast, use_container_width=True)
            else:
                st.warning("Selecione pelo menos uma região válida para comparação.")
        else:
            st.warning("Dados regionais não estão disponíveis ou nenhuma região foi selecionada.")
    
    # Rodapé da aplicação
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center;'>
            <p>Desenvolvido por Águas Guariroba | Versão 1.0.0</p>
            <p>Dados: ERA5 Reanalysis (C3S) | © 2025</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Exibir informações sobre dados ausentes
    if modules and not all(modules.values()):
        st.sidebar.markdown("---")
        st.sidebar.header("📦 Módulos Ausentes")
        
        for module, available in modules.items():
            if not available:
                st.sidebar.warning(f"❌ {module} não está instalado")
        
        st.sidebar.markdown(
            """
            Para instalar os módulos ausentes, execute:
            ```
            pip install cdsapi xarray geopandas cartopy scikit-learn
            ```
            """
        )

# Ponto de entrada da aplicação
if __name__ == "__main__":
    main()
