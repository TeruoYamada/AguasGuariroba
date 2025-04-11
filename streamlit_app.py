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
import geopandas as gpd
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Previsão de Precipitação - Campo Grande (Área Urbana)")

# Função para carregar as credenciais do CDS API
try:
    # Tente carregar a partir de secrets.toml
    cds_url = st.secrets.get("cds", {}).get("url", "https://cds.climate.copernicus.eu/api/v2")
    cds_key = st.secrets.get("cds", {}).get("key", None)
    
    # Se não houver chave nos secrets, tente usar variáveis de ambiente
    if not cds_key:
        import os
        cds_key = os.environ.get("CDS_KEY", None)
        
    # Se ainda não houver chave, permita entrada manual
    if not cds_key:
        if 'cds_key_input' not in st.session_state:
            st.session_state.cds_key_input = ''
            
        st.sidebar.markdown("### Credenciais do CDS API")
        cds_key = st.sidebar.text_input("CDS API Key:", value=st.session_state.cds_key_input, type="password")
        st.session_state.cds_key_input = cds_key
        
        if not cds_key:
            st.warning("Por favor, insira sua chave API do Copernicus Climate Data Store")
            st.stop()
            
    # Inicializar cliente CDS API
    client = cdsapi.Client(url=cds_url, key=cds_key)
    
except Exception as e:
    st.error(f"❌ Erro ao configurar o cliente CDS API: {str(e)}")
    st.warning("Verifique suas credenciais ou configure o arquivo secrets.toml adequadamente.")
    st.stop()

# Definição da área urbana de Campo Grande
# Coordenadas aproximadas do centro e bairros de Campo Grande
campo_grande_areas = {
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

# Variáveis de precipitação disponíveis no ERA5
precipitation_variables = {
    "total_precipitation": "Precipitação Total (mm)",
    "large_scale_precipitation": "Precipitação de Grande Escala (mm)",
    "convective_precipitation": "Precipitação Convectiva (mm)"
}

# Títulos e introdução
st.title("🌧️ Monitoramento e Previsão de Precipitação - Campo Grande (Área Urbana)")
st.markdown("""
Este aplicativo permite visualizar e analisar dados de precipitação para a área urbana de Campo Grande, MS. 
Os dados são obtidos do Copernicus Climate Data Store (CDS), usando o dataset ERA5.
""")

# Barra lateral para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de região da cidade
st.sidebar.subheader("Localização")
area = st.sidebar.selectbox("Selecione a região da cidade", list(campo_grande_areas.keys()))

# Determinar coordenadas da região selecionada
lat_center, lon_center = campo_grande_areas[area]

# Configurações de data e hora
st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input("Data de Início", datetime.today())
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=5))

all_hours = list(range(0, 24, 3))
start_hour = st.sidebar.selectbox("Horário Inicial", all_hours, format_func=lambda x: f"{x:02d}:00")
end_hour = st.sidebar.selectbox("Horário Final", all_hours, index=len(all_hours)-1, format_func=lambda x: f"{x:02d}:00")

# Seleção da variável de precipitação
st.sidebar.subheader("Variável de Precipitação")
precip_var = st.sidebar.selectbox(
    "Selecione a variável", 
    list(precipitation_variables.keys()),
    format_func=lambda x: precipitation_variables[x]
)

# Opções avançadas
st.sidebar.subheader("Opções Avançadas")
with st.sidebar.expander("Configurações da Visualização"):
    map_width = st.slider("Largura do Mapa (graus)", 0.1, 2.0, 0.3, step=0.1)
    animation_speed = st.slider("Velocidade da Animação (ms)", 200, 1000, 500)
    colormap = st.selectbox("Paleta de Cores", 
                           ["Blues", "viridis", "plasma", "RdYlBu_r", "gist_earth"])
    
    # Opções de projeção do produto
    product_type = st.radio(
        "Tipo de Produto",
        ["reanalysis", "ensemble_mean"],
        index=0,
        help="Reanalysis: dados históricos processados; Ensemble mean: média das previsões"
    )

# Função para extrair valores de precipitação para um ponto específico
def extract_point_timeseries(ds, lat, lon, var_name='total_precipitation'):
    """Extrai série temporal de um ponto específico do dataset."""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    # Identificar a dimensão temporal
    time_dim = 'time' if 'time' in ds.dims else 'forecast_time' if 'forecast_time' in ds.dims else None
    
    if not time_dim:
        return pd.DataFrame(columns=['time', 'precipitation'])
    
    # Extrair valores para todas as horas disponíveis
    times = []
    values = []
    
    for t_idx in range(len(ds[time_dim])):
        try:
            # Para lidar com diferentes estruturas de dados
            if len(ds[var_name].dims) > 2:  # Tem dimensão de tempo
                value = float(ds[var_name].isel({time_dim: t_idx, 'latitude': lat_idx, 'longitude': lon_idx}).values)
            else:  # Não tem dimensão de tempo (um único passo de tempo)
                value = float(ds[var_name].isel({'latitude': lat_idx, 'longitude': lon_idx}).values)
                
            times.append(pd.to_datetime(ds[time_dim].isel({time_dim: t_idx}).values))
            values.append(value)
        except Exception as e:
            st.warning(f"Erro ao extrair valor para o tempo {t_idx}: {str(e)}")
            continue
    
    # Criar DataFrame
    if times and values:
        df = pd.DataFrame({'time': times, 'precipitation': values})
        
        # Converter unidades se necessário (de m para mm)
        if var_name in ['total_precipitation', 'large_scale_precipitation', 'convective_precipitation']:
            df['precipitation'] = df['precipitation'] * 1000  # m para mm
            
        df = df.sort_values('time').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=['time', 'precipitation'])

# Função para prever valores futuros de precipitação
def predict_future_precipitation(df, days=3):
    """Gera uma previsão simples de precipitação baseada nos dados históricos."""
    if len(df) < 3:  # Precisa de pelo menos 3 pontos para uma previsão mínima
        return pd.DataFrame(columns=['time', 'precipitation', 'type'])
    
    # Preparar dados para regressão
    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
    
    # Modelo de regressão linear simples - usamos para tendência geral
    X = df_hist['time_numeric'].values.reshape(-1, 1)
    y = df_hist['precipitation'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Gerar pontos futuros
    last_time = df_hist['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]  # 4 pontos por dia (6h)
    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
    
    # Prever valores
    future_precip = model.predict(np.array(future_time_numeric).reshape(-1, 1))
    
    # Adicionar componente sazonal (ciclo diário simples)
    for i, t in enumerate(future_times):
        hour = t.hour
        # Ciclo diário: mais chuva à tarde
        hour_factor = 1.0 + 0.5 * np.sin(np.pi * (hour - 6) / 12)
        future_precip[i] *= hour_factor
    
    # Limitar valores previstos (precipitação não pode ser negativa)
    future_precip = np.maximum(future_precip, 0)
    
    # Criar DataFrame com previsão
    df_pred = pd.DataFrame({
        'time': future_times,
        'precipitation': future_precip,
        'type': 'forecast'
    })
    
    # Adicionar indicador aos dados históricos
    df_hist['type'] = 'historical'
    
    # Combinar histórico e previsão
    result = pd.concat([df_hist[['time', 'precipitation', 'type']], df_pred], ignore_index=True)
    return result

# Função para calcular acumulado diário de precipitação
def calculate_daily_precipitation(df):
    """Calcula a precipitação acumulada por dia."""
    df['date'] = df['time'].dt.date
    daily_precip = df.groupby('date')['precipitation'].sum().reset_index()
    daily_precip['precipitation'] = daily_precip['precipitation'].round(1)  # Arredondar para 1 casa decimal
    return daily_precip

# Função para calcular estatísticas de precipitação
def calculate_precipitation_stats(df):
    """Calcula estatísticas básicas de precipitação."""
    if df.empty:
        return None
    
    stats = {
        'total': df['precipitation'].sum(),
        'max': df['precipitation'].max(),
        'mean': df['precipitation'].mean(),
        'days_with_rain': (df.groupby('date')['precipitation'].sum() > 0.1).sum()
    }
    
    return stats

# Função para obter emoji baseado no valor de precipitação
def get_precipitation_emoji(value):
    """Retorna um emoji baseado no valor de precipitação."""
    if value == 0:
        return "☀️"  # Sol (sem chuva)
    elif value < 2.5:
        return "🌦️"  # Sol com nuvem de chuva (chuva fraca)
    elif value < 10:
        return "🌧️"  # Nuvem com chuva (chuva moderada)
    elif value < 25:
        return "⛈️"  # Nuvem com chuva e trovão (chuva forte)
    else:
        return "🌊"  # Onda (chuva muito forte/inundação)

# Função para categorizar intensidade de precipitação
def categorize_precipitation(value):
    """Categoriza a intensidade da precipitação."""
    if value == 0:
        return "Sem chuva", "green"
    elif value < 2.5:
        return "Chuva fraca", "#92D050"
    elif value < 10:
        return "Chuva moderada", "#FFCC00"
    elif value < 25:
        return "Chuva forte", "#FF9900"
    else:
        return "Chuva muito forte", "#FF0000"

# Criar uma função simples para adicionar principais pontos de referência de Campo Grande ao mapa
def add_campo_grande_landmarks(ax):
    """Adiciona pontos de referência de Campo Grande ao mapa."""
    landmarks = {
        "Parque dos Poderes": [-20.4407, -54.5772],
        "Shopping Campo Grande": [-20.4622, -54.5890],
        "UFMS": [-20.5008, -54.6142],
        "Aeroporto": [-20.4687, -54.6725],
        "Parque das Nações Indígenas": [-20.4532, -54.5726],
        "Terminal Rodoviário": [-20.4830, -54.6209],
        "Morenão": [-20.5114, -54.6150]
    }
    
    for name, (lat, lon) in landmarks.items():
        ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
        ax.text(lon+0.005, lat, name, transform=ccrs.PlateCarree(), fontsize=8, 
                horizontalalignment='left', verticalalignment='center')

# Função para adicionar bairros de Campo Grande ao mapa
def add_campo_grande_neighborhoods(ax):
    """Adiciona regiões de Campo Grande ao mapa."""
    for region, (lat, lon) in campo_grande_areas.items():
        # Desenha círculo para representar a região
        ax.add_patch(plt.Circle((lon, lat), 0.01, color='blue', 
                              alpha=0.2, transform=ccrs.PlateCarree()))
        
        # Adiciona nome se não for a região atualmente selecionada
        if region != area:
            ax.text(lon, lat, region, transform=ccrs.PlateCarree(), 
                   fontsize=7, ha='center', va='center')
        else:
            # Destaca a região selecionada
            ax.text(lon, lat, region, transform=ccrs.PlateCarree(), 
                   fontsize=8, ha='center', va='center', weight='bold', color='red')

# Função principal para gerar análise de precipitação
def generate_precipitation_analysis():
    """Função principal para baixar dados e gerar análise de precipitação."""
    # Selecionar o dataset adequado
    dataset = "reanalysis-era5-single-levels"
    
    # Format dates and times correctly for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Create list of hours in the correct format
    hours = []
    current_hour = start_hour
    while True:
        hours.append(f"{current_hour:02d}:00")
        if current_hour == end_hour:
            break
        current_hour = (current_hour + 3) % 24
        if current_hour == start_hour:  # Evitar loop infinito
            break
    
    # Se não tivermos horas definidas, usar padrão
    if not hours:
        hours = ['00:00', '06:00', '12:00', '18:00']
    
    # Definir área geográfica de interesse (cobrindo a área urbana de Campo Grande)
    area = [
        lat_center + map_width/2,  # North
        lon_center - map_width/2,  # West
        lat_center - map_width/2,  # South
        lon_center + map_width/2   # East
    ]
    
    # Preparar request para API
    request = {
        'product_type': product_type,
        'variable': precip_var,
        'year': [d.strftime('%Y') for d in pd.date_range(start=start_date, end=end_date, freq='D')],
        'month': [d.strftime('%m') for d in pd.date_range(start=start_date, end=end_date, freq='D').unique('month')],
        'day': [d.strftime('%d') for d in pd.date_range(start=start_date, end=end_date, freq='D')],
        'time': hours,
        'area': area,
        'format': 'netcdf'
    }
    
    filename = f'precipitation_CampoGrande_{area}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados de precipitação do CDS...'):
            client.retrieve(dataset, request, filename)
        
        # Abrir o dataset NetCDF
        ds = xr.open_dataset(filename)
        
        # Verificar variáveis disponíveis
        variable_names = list(ds.data_vars)
        
        # Se a variável selecionada não estiver disponível, usar a primeira variável
        if precip_var not in variable_names:
            precip_var_actual = variable_names[0]
            st.warning(f"A variável {precip_var} não está disponível. Usando {precip_var_actual} como alternativa.")
        else:
            precip_var_actual = precip_var
        
        # Extrair série temporal para o ponto central (área selecionada)
        with st.spinner(f"Extraindo dados de precipitação para {area} em Campo Grande..."):
            df_timeseries = extract_point_timeseries(ds, lat_center, lon_center, var_name=precip_var_actual)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair dados de precipitação para este local.")
            return None
        
        # Calcular estatísticas diárias
        df_timeseries['date'] = df_timeseries['time'].dt.date
        daily_precip = calculate_daily_precipitation(df_timeseries)
        
        # Gerar previsão para os próximos dias (se necessário)
        with st.spinner("Gerando previsão de precipitação..."):
            df_forecast = predict_future_precipitation(df_timeseries, days=5)
        
        # --- Criação da animação ---
        # Identificar frames disponíveis (passos de tempo)
        time_dim = 'time' if 'time' in ds.dims else 'forecast_time' if 'forecast_time' in ds.dims else None
        
        if not time_dim:
            st.error("Não foi possível identificar dimensão temporal nos dados.")
            return None
        
        frames = len(ds[time_dim])
        
        if frames < 1:
            st.error("Erro: Dados insuficientes para animação.")
            return None
        
        # Determinar range de cores para precipitação
        # Para precipitação, valores típicos: 0 a 25mm
        vmin = 0
        vmax = max(25, float(ds[precip_var_actual].max().values) * 1000)  # Converter m para mm
        
        # Criar figura para animação
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Adicionar features básicas
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
        
        # Adicionar grid
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Definir extensão do mapa
        ax.set_extent([lon_center - map_width/2, lon_center + map_width/2, 
                      lat_center - map_width/2, lat_center + map_width/2], 
                     crs=ccrs.PlateCarree())
        
        # Adicionar pontos de referência e bairros de Campo Grande
        add_campo_grande_landmarks(ax)
        add_campo_grande_neighborhoods(ax)
        
        # Obter primeiro frame para inicializar
        first_frame_data = ds[precip_var_actual].isel({time_dim: 0}).values
        first_frame_time = pd.to_datetime(ds[time_dim].values[0])
        
        # Converter valores de m para mm se necessário
        if precip_var_actual in ['total_precipitation', 'large_scale_precipitation', 'convective_precipitation']:
            first_frame_data = first_frame_data * 1000  # m para mm
        
        # Criar mapa de cores personalizado para precipitação
        if colormap == "Blues":
            colors = [(1, 1, 1), (0.8, 0.8, 0.95), (0.4, 0.4, 0.8), (0.2, 0.2, 0.7), (0, 0, 0.6)]
            precip_cmap = LinearSegmentedColormap.from_list("precip_blues", colors)
        else:
            precip_cmap = plt.get_cmap(colormap)
        
        # Criar mapa de cores
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data,
                          transform=ccrs.PlateCarree(),
                          cmap=precip_cmap, vmin=vmin, vmax=vmax)
        
        # Adicionar title com informações do primeiro frame
        title = ax.set_title(f"Precipitação - {area}, Campo Grande - {first_frame_time.strftime('%d/%m/%Y %H:%M')}")
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label(precipitation_variables.get(precip_var_actual, "Precipitação (mm)"))
        
        # Adicionar marcador para o ponto selecionado
        city_marker = ax.plot(lon_center, lat_center, 'ro', markersize=8, transform=ccrs.PlateCarree())[0]
        
        # Desenhar um círculo para destacar a área urbana principal
        urban_area = plt.Circle((campo_grande_areas["Centro"][1], campo_grande_areas["Centro"][0]), 
                              0.03, color='red', fill=False, transform=ccrs.PlateCarree())
        ax.add_patch(urban_area)
            
        # Função para atualizar frames da animação
        def update_frame(frame):
            # Atualizar dados
            frame_data = ds[precip_var_actual].isel({time_dim: frame}).values
            
            # Converter de m para mm se necessário
            if precip_var_actual in ['total_precipitation', 'large_scale_precipitation', 'convective_precipitation']:
                frame_data = frame_data * 1000
                
            # Atualizar plot
            im.set_array(frame_data.ravel())
            
            # Atualizar título com timestamp
            frame_time = pd.to_datetime(ds[time_dim].values[frame])
            title.set_text(f"Precipitação - {area}, Campo Grande - {frame_time.strftime('%d/%m/%Y %H:%M')}")
            
            return im, title
        
        # Criar animação
        anim = animation.FuncAnimation(fig, update_frame, frames=frames, 
                                      interval=animation_speed, blit=False)
        
        # Salvar animação como GIF (opcional)
        anim_file = f"precipitation_animation_CampoGrande_{area}.gif"
        anim.save(anim_file, writer='pillow', fps=2)
        
        return {
            'dataset': ds,
            'timeseries': df_timeseries,
            'daily': daily_precip,
            'forecast': df_forecast,
            'animation_file': anim_file,
            'figure': fig
        }
        
    except Exception as e:
        st.error(f"❌ Erro ao processar dados de precipitação: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None

# Interface principal - abas para diferentes visualizações
tabs = st.tabs(["📊 Análise de Precipitação", "🗺️ Mapa de Precipitação", "📈 Série Temporal", "ℹ️ Sobre"])

with tabs[0]:
    st.header(f"Análise de Precipitação para {area}, Campo Grande")
    
    # Botão para iniciar análise
    if st.button("🔄 Gerar Análise de Precipitação"):
        # Executar análise de precipitação
        results = generate_precipitation_analysis()
        
        if results:
            # Mostrar resumo de estatísticas
            st.subheader("📊 Resumo da Precipitação")
            
            # Layout de duas colunas
            col1, col2 = st.columns(2)
            
            with col1:
                # Tabela de precipitação diária
                st.markdown("### Precipitação Diária")
                daily = results['daily']
                
                if not daily.empty:
                    # Adicionar emojis e categorias
                    daily['emoji'] = daily['precipitation'].apply(get_precipitation_emoji)
                    daily['categoria'], daily['cor'] = zip(*daily['precipitation'].apply(categorize_precipitation))
                    
                    # Formatação para exibição
                    display_df = daily.copy()
                    display_df['date'] = display_df['date'].apply(lambda x: x.strftime('%d/%m/%Y'))
                    display_df = display_df.rename(columns={
                        'date': 'Data',
                        'precipitation': 'Precipitação (mm)',
                        'emoji': 'Símbolo',
                        'categoria': 'Categoria'
                    })
                    
                    # Exibir dataframe formatado
                    st.dataframe(display_df[['Data', 'Precipitação (mm)', 'Símbolo', 'Categoria']], 
                                use_container_width=True)
                else:
                    st.warning("Sem dados diários disponíveis.")
            
            with col2:
                # Estatísticas gerais
                stats = calculate_precipitation_stats(results['timeseries'])
                
                if stats:
                    st.markdown("### Estatísticas")
                    
                    # Métricas em cards
                    col_a, col_b = st.columns(2)
                    col_a.metric("Precipitação Total", f"{stats['total']:.1f} mm")
                    col_b.metric("Precipitação Máxima", f"{stats['max']:.1f} mm")
                    
                    col_c, col_d = st.columns(2)
                    col_c.metric("Média Diária", f"{stats['mean']:.1f} mm")
                    col_d.metric("Dias com Chuva", f"{stats['days_with_rain']}")
                    
                    # Interpretação
                    if stats['total'] == 0:
                        st.info("📋 Interpretação: Sem registro de chuvas no período.")
                    elif stats['total'] < 5:
                        st.info("📋 Interpretação: Período com chuvas fracas e isoladas.")
                    elif stats['total'] < 20:
                        st.info("📋 Interpretação: Período com chuvas moderadas.")
                    else:
                        st.info("📋 Interpretação: Período com chuvas significativas.")
            
            # Mostrar gráfico de previsão
            st.subheader("📈 Previsão de Precipitação")
            
            forecast_df = results['forecast']
            if not forecast_df.empty:
                # Criar figura para o gráfico
                fig_forecast = plt.figure(figsize=(10, 6))
                ax = fig_forecast.add_subplot(111)
                
                # Separar dados históricos e previsão
                hist_data = forecast_df[forecast_df['type'] == 'historical']
                pred_data = forecast_df[forecast_df['type'] == 'forecast']
                
                # Plotar dados históricos
                ax.plot(hist_data['time'], hist_data['precipitation'], 
                       'o-', color='blue', label='Dados históricos')
                
                # Plotar previsão
                ax.plot(pred_data['time'], pred_data['precipitation'], 
                       '--', color='red', label='Previsão')
                
                # Adicionar áreas sombreadas para períodos do dia
                days = pd.date_range(start=forecast_df['time'].min().date(), 
                                     end=forecast_df['time'].max().date() + timedelta(days=1), 
                                     freq='D')
                
                for day in days:
                    # Manhã (6-12h) - amarelo claro
                    morning_start = pd.Timestamp(day.year, day.month, day.day, 6)
                    morning_end = pd.Timestamp(day.year, day.month, day.day, 12)
                    ax.axvspan(morning_start, morning_end, alpha=0.1, color='yellow')
                    
                    # Tarde (12-18h) - laranja claro
                    afternoon_start = pd.Timestamp(day.year, day.month, day.day, 12)
                    afternoon_end = pd.Timestamp(day.year, day.month, day.day, 18)
                    ax.axvspan(afternoon_start, afternoon_end = pd.Timestamp(day.year, day.month, day.day, 18))
                    ax.axvspan(afternoon_start, afternoon_end, alpha=0.1, color='orange')
                    
                    # Noite (18-6h) - azul claro
                    night_start = pd.Timestamp(day.year, day.month, day.day, 18)
                    night_end = pd.Timestamp(day.year, day.month, day.day + 1, 6)
                    ax.axvspan(night_start, night_end, alpha=0.1, color='blue')
                
                # Formatar eixos
                ax.set_xlabel('Data e Hora')
                ax.set_ylabel('Precipitação (mm)')
                ax.set_title('Previsão de Precipitação para ' + area)
                ax.legend()
                
                # Formatar datas no eixo x
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                fig_forecast.autofmt_xdate()
                
                st.pyplot(fig_forecast)
                
                # Explicação da previsão
                st.markdown("""
                **Observações sobre a previsão:**
                - A previsão é baseada em uma extrapolação simples dos dados históricos
                - Áreas coloridas representam períodos do dia (manhã, tarde, noite)
                - A precisão diminui para períodos mais distantes
                """)
            else:
                st.warning("Não foi possível gerar previsão com os dados disponíveis.")

with tabs[1]:
    st.header("🗺️ Mapa de Precipitação")
    
    if 'results' in locals():
        # Mostrar animação
        st.subheader("Animação da Precipitação")
        
        # Exibir animação
        st.image(results['animation_file'], use_column_width=True)
        
        # Mostrar mapa estático do último frame
        st.subheader("Última Atualização")
        st.pyplot(results['figure'])
    else:
        st.info("Execute a análise na aba '📊 Análise de Precipitação' para visualizar os mapas.")

with tabs[2]:
    st.header("📈 Série Temporal de Precipitação")
    
    if 'results' in locals():
        # Gráfico de série temporal
        fig_ts = plt.figure(figsize=(10, 6))
        ax = fig_ts.add_subplot(111)
        
        # Plotar dados
        ax.plot(results['timeseries']['time'], 
               results['timeseries']['precipitation'], 
               'b-', label='Precipitação')
        
        # Adicionar média móvel de 6h
        window_size = 2  # 2 pontos para média de 6h (considerando dados de 3 em 3h)
        if len(results['timeseries']) >= window_size:
            rolling_mean = results['timeseries']['precipitation'].rolling(window=window_size).mean()
            ax.plot(results['timeseries']['time'], rolling_mean, 
                   'r--', label=f'Média móvel {window_size*3}h')
        
        # Configurar gráfico
        ax.set_xlabel('Data e Hora')
        ax.set_ylabel('Precipitação (mm)')
        ax.set_title(f'Série Temporal de Precipitação - {area}')
        ax.legend()
        
        # Formatar datas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        fig_ts.autofmt_xdate()
        
        st.pyplot(fig_ts)
        
        # Opção para download dos dados
        st.download_button(
            label="📥 Baixar Dados de Precipitação",
            data=results['timeseries'].to_csv(index=False).encode('utf-8'),
            file_name=f'precipitacao_{area}.csv',
            mime='text/csv'
        )
    else:
        st.info("Execute a análise na aba '📊 Análise de Precipitação' para visualizar a série temporal.")

with tabs[3]:
    st.header("ℹ️ Sobre este Aplicativo")
    
    st.markdown("""
    ### Monitoramento de Precipitação - Campo Grande/MS
    
    **Objetivo:**
    Este aplicativo fornece visualizações e análises de dados de precipitação para a área urbana de Campo Grande, Mato Grosso do Sul.
    
    **Fonte dos Dados:**
    - Dados meteorológicos do ERA5 (Copernicus Climate Data Store)
    - Reanálise e previsões de precipitação
    
    **Funcionalidades:**
    - Visualização espacial da precipitação
    - Análise temporal por região da cidade
    - Previsão simples baseada em tendências
    - Estatísticas de precipitação
    
    **Desenvolvido por:**
    [Teruo Yamada]
    
    **Contato:**
    [eng.teruoyamada@hotmail.com]
    
    **Última Atualização:**
    {}
    """.format(datetime.now().strftime('%d/%m/%Y')))
    
    st.markdown("---")
    st.markdown("""
    **Aviso Legal:**
    As previsões apresentadas são baseadas em modelos estatísticos simplificados e não substituem as previsões oficiais do órgãos meteorológicos.
    """)

# Rodapé
st.markdown("---")
st.markdown("""
<small>Desenvolvido com Python, Streamlit, Cartopy e CDS API | Dados do Copernicus Climate Data Store</small>
""", unsafe_allow_html=True)

# Limpeza de arquivos temporários
if 'results' in locals():
    try:
        os.remove(results['animation_file'])
        os.remove(filename)
    except:
        pass

