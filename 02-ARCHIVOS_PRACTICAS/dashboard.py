import streamlit as st
import pandas as pd
import sqlite3 # Necesario para la conexión inicial para PRAGMA
import datetime

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Panel de Sensores de Empresa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE_PATH = 'company_sensors.db'

# --- Conexión a la Base de Datos y Funciones de Carga ---
try:
    conn_raw_check = sqlite3.connect(f"file:{DB_FILE_PATH}?mode=rw", uri=True)
    conn_raw_check.execute('PRAGMA journal_mode=WAL;')
    conn_raw_check.close()
except sqlite3.OperationalError:
    st.sidebar.error(
        f"La base de datos '{DB_FILE_PATH}' no existe o no se puede abrir. "
        "Por favor, ejecuta primero el script 'data_generator.py'."
    )
    st.error(f"Error Crítico: La base de datos '{DB_FILE_PATH}' no se encuentra. "
             "Asegúrate de que 'data_generator.py' se haya ejecutado y creado el archivo.")
    st.stop()
except sqlite3.Error as e:
    st.sidebar.error(f"Error al configurar WAL: {e}")
    st.sidebar.warning("El dashboard podría tener problemas de bloqueo.")

def get_db_conn():
    return st.connection('sensor_db', type='sql', url=f"sqlite:///{DB_FILE_PATH}")

# --- Barra Lateral con Filtros ---
with st.sidebar:
    st.header("Filtros y Controles")
    
    try:
        conn_temp = get_db_conn()
        distinct_sensor_types_df = conn_temp.query("SELECT DISTINCT sensor_type FROM sensor_data ORDER BY sensor_type", ttl=300)
        sensor_type_list = distinct_sensor_types_df['sensor_type'].tolist() if not distinct_sensor_types_df.empty else []
        sensor_type_options = ["Todos"] + sensor_type_list
    except Exception as e:
        st.sidebar.error(f"No se pudieron cargar los tipos de sensor: {e}")
        sensor_type_options = ["Todos"]

    selected_sensor_type = st.selectbox(
        "Tipo de Sensor:",
        options=sensor_type_options,
        key="selected_sensor_type_key" 
    )

    refresh_interval = st.slider(
        "Intervalo de Actualización (segundos):",
        min_value=2, max_value=60, value=5, step=1,
        key="refresh_interval_slider"
    )
    st.caption(f"El panel intentará actualizarse cada {st.session_state.get('refresh_interval_slider', 5)} segundos.")

# --- Título Principal ---
st.title("📊 Panel de Control de Sensores de Empresa en Tiempo Real")
status_placeholder = st.empty()

# --- Funciones de Visualización ---
def display_humidity_chart(df_filtered):
    st.subheader("💧 Humedad (%)")
    humidity_data = df_filtered[df_filtered['sensor_type'] == 'humidity']
    if not humidity_data.empty and 'reading_float' in humidity_data.columns and not humidity_data['reading_float'].isnull().all():
        humidity_data_chart = humidity_data.copy()
        humidity_data_chart['timestamp'] = pd.to_datetime(humidity_data_chart['timestamp'])
        st.line_chart(humidity_data_chart.set_index('timestamp')['reading_float'], use_container_width=True)
    else:
        st.caption("No hay datos de humedad para mostrar con los filtros actuales.")

def display_fire_status(df_all_sensors_for_fire):
    st.subheader("🔥 Estado de Detección de Incendio")
    if not df_all_sensors_for_fire.empty:
        latest_reading_series = df_all_sensors_for_fire.iloc[0]
        latest_status = latest_reading_series['reading_text']
        timestamp_val = latest_reading_series['timestamp']
        sensor_id_val = latest_reading_series['sensor_id']
        
        timestamp_str = timestamp_val.strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notnull(timestamp_val) else "N/A"
        
        if latest_status == "Fuego Detectado":
            st.error(f"🚨 ¡FUEGO DETECTADO! ({sensor_id_val}) 🚨", icon="🔥")
            st.metric(label=f"Sensor: {sensor_id_val} ({timestamp_str})", value="ALERTA CRÍTICA", delta="Revisar Inmediatamente", delta_color="inverse")
        elif latest_status == "Alerta Leve":
            st.warning(f"⚠️ ALERTA LEVE DE HUMO ({sensor_id_val}) ⚠️", icon="💨")
            st.metric(label=f"Sensor: {sensor_id_val} ({timestamp_str})", value="Advertencia", delta="Posible Incidente", delta_color="normal")
        else: # "Normal"
            st.success(f"✔️ Sistema Normal ({sensor_id_val})", icon="✅")
            st.metric(label=f"Sensor: {sensor_id_val} ({timestamp_str})", value="Normal", delta="Todo Despejado", delta_color="off")
    else:
        st.info("No hay datos recientes de detección de incendio.")

def display_motion_data(df_filtered):
    st.subheader("🏃 Movimiento de Operadores")
    motion_data = df_filtered[df_filtered['sensor_type'] == 'operator_movement'].copy()
    
    if not motion_data.empty and 'reading_int' in motion_data.columns and not motion_data['reading_int'].isnull().all():
        motion_data['timestamp'] = pd.to_datetime(motion_data['timestamp'])
        st.caption("Eventos de movimiento recientes (últimos 5):")
        st.dataframe(
            motion_data[['timestamp', 'sensor_id', 'reading_int']].rename(columns={'reading_int': 'event_count'}).head(5),
            use_container_width=True, 
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Hora", format="HH:mm:ss", timezone="UTC"),
                "sensor_id": "ID Sensor",
                "event_count": "Conteo Eventos"
            }
        )
        
        if not motion_data.empty:
            total_motion_by_sensor = motion_data.groupby('sensor_id')['reading_int'].sum().reset_index()
            if not total_motion_by_sensor.empty:
                st.caption("Conteo total de eventos de movimiento por sensor (en datos cargados):")
                st.bar_chart(total_motion_by_sensor.set_index('sensor_id')['reading_int'], use_container_width=True)
    else:
        st.caption("No hay datos de movimiento para mostrar con los filtros actuales.")

def display_general_data_table(df_filtered, title_suffix=""):
    st.subheader(f"📜 Registros Recientes de Sensores{title_suffix} (hasta 10)")
    if not df_filtered.empty:
        df_display = df_filtered.copy()
        st.dataframe(
            df_display.head(10), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Marca de Tiempo", format="YYYY-MM-DD HH:mm:ss", timezone="UTC"),
                "sensor_id": "ID Sensor",
                "sensor_type": "Tipo Sensor",
                "reading_float": st.column_config.NumberColumn("Valor (Float)", format="%.2f"),
                "reading_text": "Valor (Text)",
                "reading_int": "Valor (Int)"
            })
    else:
        st.info(f"No hay datos de sensores para mostrar con los filtros actuales{title_suffix}.")

def display_key_stats(df_live):
    if not df_live.empty:
        st.subheader("Estadísticas Rápidas")
        stat_col1, stat_col2, stat_col3 = st.columns(3)

        # Estadísticas de Humedad
        humidity_df = df_live[df_live['sensor_type'] == 'humidity']
        if not humidity_df.empty:
            humidity_stats = humidity_df['reading_float'].agg(['min', 'max', 'mean'])
            with stat_col1:
                st.metric("Humedad Mín.", f"{humidity_stats['min']:.1f}%" if pd.notnull(humidity_stats['min']) else "N/A")
                st.metric("Humedad Máx.", f"{humidity_stats['max']:.1f}%" if pd.notnull(humidity_stats['max']) else "N/A")
                st.metric("Humedad Prom.", f"{humidity_stats['mean']:.1f}%" if pd.notnull(humidity_stats['mean']) else "N/A")
        else:
            with stat_col1:
                st.caption("Sin datos de humedad para estadísticas.")

        # Estadísticas de Movimiento
        motion_df = df_live[df_live['sensor_type'] == 'operator_movement']
        if not motion_df.empty:
            motion_stats = motion_df['reading_int'].agg(['min', 'max', 'mean', 'sum'])
            with stat_col2:
                st.metric("Mov. Mín. Eventos", f"{motion_stats['min']:.0f}" if pd.notnull(motion_stats['min']) else "N/A")
                st.metric("Mov. Máx. Eventos", f"{motion_stats['max']:.0f}" if pd.notnull(motion_stats['max']) else "N/A")
                st.metric("Mov. Prom. Eventos", f"{motion_stats['mean']:.1f}" if pd.notnull(motion_stats['mean']) else "N/A")
                st.metric("Mov. Total Eventos", f"{motion_stats['sum']:.0f}" if pd.notnull(motion_stats['sum']) else "N/A")
        else:
            with stat_col2:
                st.caption("Sin datos de movimiento para estadísticas.")
        
        # Conteo de Alertas de Incendio
        fire_alerts_df = df_live[
            (df_live['sensor_type'] == 'fire_detection') & 
            (df_live['reading_text'].isin(["Fuego Detectado", "Alerta Leve"]))
        ]
        num_fire_alerts = len(fire_alerts_df)
        with stat_col3:
            st.metric("Total Alertas de Incendio", num_fire_alerts)
            if num_fire_alerts > 0:
                last_alert_time = fire_alerts_df['timestamp'].max()
                st.caption(f"Última alerta: {last_alert_time.strftime('%H:%M:%S') if pd.notnull(last_alert_time) else 'N/A'}")
            else:
                st.caption("Ninguna alerta de incendio reciente.")
        st.divider()

def display_latest_per_sensor(df_live):
    if not df_live.empty:
        st.subheader("Última Lectura por Sensor ID (de datos cargados en esta actualización)")
        latest_readings_per_sensor = df_live.loc[df_live.groupby('sensor_id')['timestamp'].idxmax()]
        
        st.dataframe(
            latest_readings_per_sensor[['sensor_id', 'sensor_type', 'timestamp', 'reading_float', 'reading_text', 'reading_int']].sort_values(by="timestamp", ascending=False),
            use_container_width=True, 
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Última Hora", format="YYYY-MM-DD HH:mm:ss", timezone="UTC"),
                "sensor_id": "ID Sensor",
                "sensor_type": "Tipo de Sensor",
                "reading_float": st.column_config.NumberColumn("Valor (Float)", format="%.2f"),
                "reading_text": "Valor (Text)",
                "reading_int": "Valor (Int)"
            }
        )
        st.divider()

# --- Fragmento para Actualización Automática ---
@st._fragment(run_every=st.session_state.get('refresh_interval_slider', 5))
def auto_refresh_dashboard_fragment():
    current_refresh_interval = st.session_state.get('refresh_interval_slider', 5)
    status_placeholder.caption(f"Actualizando datos (intervalo: {current_refresh_interval}s)... Última: {datetime.datetime.now().strftime('%H:%M:%S')}")

    query_base = "SELECT id, timestamp, sensor_id, sensor_type, reading_float, reading_text, reading_int FROM sensor_data"
    conditions = []
    params_list = []

    current_selected_sensor_type = st.session_state.get("selected_sensor_type_key", "Todos")
    

    if conditions:
        query_base += " WHERE " + " AND ".join(conditions)
    
    query_base += " ORDER BY timestamp DESC LIMIT 200" # Trae los 200 más recientes que CUMPLAN el filtro SQL

    conn = get_db_conn()
    
    params_for_query = None
    if params_list:
        params_for_query = [tuple(params_list)] 

    try:
        df_live = conn.query(query_base, params=params_for_query, ttl=0) 
        if 'timestamp' in df_live.columns:
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
    except Exception as e:
        st.error(f"Error al cargar datos en el fragmento: {e}")
        # st.exception(e) 
        df_live = pd.DataFrame() 

    if current_selected_sensor_type != "Todos":
        df_live = df_live[df_live['sensor_type'] == current_selected_sensor_type]
    
    # --- Presentación de Información Adicional ---
    display_key_stats(df_live)
    display_latest_per_sensor(df_live)

    # --- Visualizaciones Principales ---
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        display_humidity_chart(df_live) # df_live ya está filtrado por sensor_type si no es "Todos"
        display_motion_data(df_live)    # df_live ya está filtrado por sensor_type si no es "Todos"
        
    with col2:
        # Para el estado de incendio, necesitamos los datos de 'fire_detection'.
        # Si el filtro principal ya es 'fire_detection', df_live ya los contiene.
        # Si el filtro es "Todos", necesitamos filtrar df_live.
        fire_data_for_status = df_live[df_live['sensor_type'] == 'fire_detection'] if current_selected_sensor_type == "Todos" else df_live
        display_fire_status(fire_data_for_status) 

    # La tabla general mostrará los datos según el filtro SQL principal.
    display_general_data_table(df_live, title_suffix=f" ({current_selected_sensor_type})" if current_selected_sensor_type != "Todos" else " (Todos los Tipos)")
    
    status_placeholder.caption(f"Panel actualizado a las: {datetime.datetime.now().strftime('%H:%M:%S')} (Intervalo: {current_refresh_interval}s)")

# Llamar al fragmento para que se ejecute y se actualice periódicamente.
auto_refresh_dashboard_fragment()