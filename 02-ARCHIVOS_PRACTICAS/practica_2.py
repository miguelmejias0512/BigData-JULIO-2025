import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Explorador de Modelos Predictivos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carga y Cacheo de Datos ---
@st.cache_data
def load_data():
    """
    Carga el dataset de California Housing, lo divide en conjuntos de entrenamiento y prueba,
    y escala las características.
    """
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    
    # Scikit-learn a partir de la versión 1.2 ya no incluye lat y lon en .data
    # Esta lógica asegura que se asignen correctamente.
    df['Latitude'] = housing.data[:, -2] if 'Latitude' not in housing.feature_names else df['Latitude']
    df['Longitude'] = housing.data[:, -1] if 'Longitude' not in housing.feature_names else df['Longitude']

    # Asegurar que las características (X) no incluyan latitud, longitud ni el target.
    feature_names_to_use = [name for name in housing.feature_names if name not in ['Latitude', 'Longitude']]
    X = df[feature_names_to_use]
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar características es una buena práctica para muchos modelos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Devolvemos tanto los datos escalados como los no escalados y el scaler
    return df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names_to_use

# --- Entrenamiento de Modelos (Cacheado) ---
@st.cache_resource
def train_model(model_name, X_train, y_train, params):
    """
    Entrena un modelo predictivo con los hiperparámetros dados.
    Utiliza st.cache_resource para evitar re-entrenar en cada rerun.
    """
    if model_name == 'Regresión Lineal':
        model = LinearRegression()
    elif model_name == 'Árbol de Decisión':
        model = DecisionTreeRegressor(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], max_depth=params['max_depth'], random_state=42)
    elif model_name == 'Support Vector Regressor (SVR)':
        model = SVR(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])

    elif model_name == 'Support Vector Regressor (SVR)':
        model = SVR(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
    
    # NUEVO BLOQUE PARA LA RED NEURONAL
    elif model_name == 'Red Neuronal (MLP)':
        # Definir la arquitectura del modelo
        model = Sequential([
            Input(shape=(X_train.shape[1],)), # Capa de entrada con el número de características
            Dense(params['neurons_layer1'], activation='relu'), # 1ra Capa Oculta
            Dense(params['neurons_layer2'], activation='relu'), # 2da Capa Oculta
            Dense(1) # Capa de salida con 1 neurona para regresión
        ])
        
        # Compilar el modelo
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Entrenar el modelo (aquí se realiza el entrenamiento real)
        # Nota: El return de la función ya hace el .fit(), pero en un caso real de Keras,
        # el .fit se llamaría aquí y se devolvería el modelo entrenado.
        # Por ahora, solo definimos y compilamos el modelo. El .fit se hará fuera.
        
        # Entrenar el modelo (El .fit para Keras es diferente)
        # Para integrarlo sin cambiar mucho la estructura, el .fit se llamará aquí
        # y devolveremos el modelo entrenado.
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=32, verbose=0)
        return model # Devuelve el modelo ya entrenado

    # El model.fit() que estaba fuera del if/elif ya no es necesario para Keras.
    # Vamos a reestructurar ligeramente para que funcione para todos.
    # if model_name != 'Red Neuronal (MLP)':
    #     model.fit(X_train, y_train)

# ... (código existente para otros modelos) ...
    elif model_name == 'Support Vector Regressor (SVR)':
        model = SVR(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
    
    # NUEVO BLOQUE PARA LA RED NEURONAL
    elif model_name == 'Red Neuronal (MLP)':
        # Definir la arquitectura del modelo
        model = Sequential([
            Input(shape=(X_train.shape[1],)), # Capa de entrada con el número de características
            Dense(params['neurons_layer1'], activation='relu'), # 1ra Capa Oculta
            Dense(params['neurons_layer2'], activation='relu'), # 2da Capa Oculta
            Dense(1) # Capa de salida con 1 neurona para regresión
        ])
        
        # Compilar el modelo
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Entrenar el modelo (aquí se realiza el entrenamiento real)
        # Nota: El return de la función ya hace el .fit(), pero en un caso real de Keras,
        # el .fit se llamaría aquí y se devolvería el modelo entrenado.
        # Por ahora, solo definimos y compilamos el modelo. El .fit se hará fuera.
        
        # Entrenar el modelo (El .fit para Keras es diferente)
        # Para integrarlo sin cambiar mucho la estructura, el .fit se llamará aquí
        # y devolveremos el modelo entrenado.
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=32, verbose=0)
        return model # Devuelve el modelo ya entrenado

    # El model.fit() que estaba fuera del if/elif ya no es necesario para Keras.
    # Vamos a reestructurar ligeramente para que funcione para todos.
    if model_name != 'Red Neuronal (MLP)':
        model.fit(X_train, y_train)

#    model.fit(X_train, y_train)
    return model


# --- Cargar todos los datos al inicio ---
df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = load_data()


# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Panel de Navegación")
app_mode = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Introducción", "Análisis Exploratorio de Datos (EDA)", "Regresión Lineal", 
     "Árbol de Decisión", "Random Forest", "Gradient Boosting", 
     "Support Vector Regressor (SVR)", "Red Neuronal (MLP)"] # <-- AÑADIR AQUÍ
)

st.sidebar.divider()

# CORRECCIÓN: Usar session_state para simular un modal y evitar el error de atributo.
# 1. Inicializar el estado si no existe.
if 'show_info_modal' not in st.session_state:
    st.session_state.show_info_modal = False

# 2. Botón para ABRIR la guía teórica.
if st.sidebar.button("Consultar info de Modelos", use_container_width=True):
    st.session_state.show_info_modal = True


# --- Contenedor de Información Teórica (simula un modal) ---
# 3. Mostrar el contenido si el estado es True.
if st.session_state.show_info_modal:
    # Usamos un st.expander como un contenedor que se puede "cerrar".
    # Se mostrará en la parte superior del área principal.
    with st.expander("Guía Teórica de Modelos Predictivos e Hiperparámetros", expanded=True):
        st.markdown("""
        ## Guía Teórica de Modelos e Hiperparámetros

        Esta guía te ayudará a entender los conceptos clave detrás de los modelos utilizados en esta aplicación.

        ---
        
        ### ¿Qué es un Hiperparámetro?
        
        En Machine Learning, un **hiperparámetro** es una configuración externa al modelo cuyo valor no se puede aprender de los datos. Es una perilla que el científico de datos ajusta *antes* de entrenar el modelo para controlar su comportamiento y rendimiento. La elección correcta de hiperparámetros es crucial y a menudo requiere experimentación.

        ---

        ### 1. Regresión Lineal

        - **Teoría:** Es el modelo más simple. Busca encontrar la mejor línea recta (o hiperplano en más dimensiones) que se ajuste a los datos. Intenta modelar la relación entre las variables de entrada (X) y la variable de salida (y) mediante una ecuación lineal.
        - **Fórmula:** $$ y = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n + \\epsilon $$
        - **Hiperparámetros en esta app:** Ninguno. La Regresión Lineal estándar no tiene hiperparámetros significativos que ajustar, lo que la hace un excelente punto de partida (baseline).

        ---

        ### 2. Árbol de Decisión

        - **Teoría:** Funciona creando un modelo similar a un árbol de reglas de "si-entonces" que segmentan los datos. En cada nodo, el árbol elige la mejor característica para dividir los datos y así minimizar el error en las predicciones.
        - **Hiperparámetros en esta app:**
            - `max_depth` (Profundidad Máxima): Controla cuán profundo puede crecer el árbol. Un valor bajo puede causar subajuste (modelo demasiado simple), mientras que un valor muy alto puede causar sobreajuste (el modelo memoriza los datos).
            - `min_samples_split` (Mínimo de Muestras para Dividir): Es el número mínimo de muestras que un nodo debe tener para poder ser dividido. Ayuda a prevenir el sobreajuste al no permitir divisiones en nodos con muy pocos datos.

        ---

        ### 3. Random Forest (Bosque Aleatorio)

        - **Teoría:** Es un método de *ensamble* que mejora la precisión y controla el sobreajuste al entrenar múltiples Árboles de Decisión. Cada árbol se entrena con una muestra aleatoria de los datos y de las características. La predicción final es el promedio de las predicciones de todos los árboles individuales.
        - **Hiperparámetros en esta app:**
            - `n_estimators` (Número de Árboles): La cantidad de árboles que se construirán en el bosque. Generalmente, más árboles mejoran el rendimiento, pero a un costo computacional mayor.
            - `max_depth`: Al igual que en un árbol de decisión individual, controla la profundidad máxima de cada árbol en el bosque.

        ---

        ### 4. Gradient Boosting

        - **Teoría:** Es otro método de *ensamble* que construye modelos (generalmente árboles) de forma secuencial. A diferencia de Random Forest, cada nuevo árbol se entrena para corregir los errores cometidos por los árboles anteriores. Es un método muy potente que a menudo resulta en modelos de alto rendimiento.
        - **Hiperparámetros en esta app:**
            - `n_estimators`: El número de árboles (etapas de boosting) que se realizarán.
            - `learning_rate` (Tasa de Aprendizaje): Reduce la contribución de cada árbol. Valores más bajos requieren más árboles para el mismo rendimiento, pero a menudo resultan en una mejor generalización. Su valor está entre 0.0 y 1.0.
            - `max_depth`: La profundidad máxima de cada árbol individual.

        ---

        ### 5. Support Vector Regressor (SVR)

        - **Teoría:** A diferencia de otros modelos que intentan minimizar el error, SVR intenta ajustar la mayor cantidad de datos posible *dentro* de un margen o "tubo" definido por un hiperparámetro épsilon (no ajustable en esta app). Los puntos fuera de este tubo son penalizados. Es muy efectivo en espacios de características de alta dimensión.
        - **Hiperparámetros en esta app:**
            - `C` (Parámetro de Regularización): Controla la penalización por los puntos que quedan fuera del margen. Un valor alto de `C` intenta ajustar más datos correctamente (menos errores), arriesgándose a un sobreajuste. Un valor bajo permite más errores (un margen más "suave").
            - `kernel`: Define la función utilizada para transformar los datos a una dimensión superior y encontrar la mejor relación.
                - `linear`: Para relaciones lineales.
                - `poly`: Para relaciones polinómicas.
                - `rbf` (Radial Basis Function): Muy popular y flexible para relaciones no lineales complejas. Su fórmula es: $$ K(x_i, x_j) = \\exp(-\\gamma ||x_i - x_j||^2) $$
            - `gamma`: Un parámetro del kernel `rbf`. Define cuánta influencia tiene un solo ejemplo de entrenamiento. Valores bajos significan una influencia lejana, mientras que valores altos significan una influencia cercana.

        ---

        ### 6. Red Neuronal (Perceptrón Multicapa - MLP)

        - **Teoría:** Inspirada en el cerebro humano, una red neuronal aprende a reconocer patrones complejos en los datos a través de capas de "neuronas" interconectadas. Es extremadamente potente y flexible, capaz de modelar relaciones no lineales muy complejas. Para regresión, la red ajusta sus conexiones internas (pesos) para minimizar el error entre los precios predichos y los reales.
        - **Hiperparámetros en esta app:**
            - `learning_rate` (Tasa de Aprendizaje): Controla cuán grandes son los ajustes a los pesos del modelo durante el entrenamiento. Es un valor pequeño y crucial para una convergencia estable.
            - `epochs` (Épocas): El número de veces que el modelo verá el conjunto de datos de entrenamiento completo. Más épocas permiten un mayor aprendizaje, pero también aumentan el riesgo de sobreajuste.
            - `Neuronas en Capa Oculta 1 / 2`: Define el número de nodos de cómputo en cada una de las dos capas ocultas. Más neuronas permiten al modelo aprender patrones más complejos, pero aumentan la complejidad y el riesgo de sobreajuste.

        """)
        # 4. Botón para CERRAR la guía.
        if st.button("Cerrar Guía"):
            st.session_state.show_info_modal = False
            st.rerun() # Fuerza un rerun para que el expander desaparezca inmediatamente.

# --- Lógica de Páginas ---
if app_mode == "Introducción":
    st.title("🤖 Explorador Interactivo de Modelos Predictivos")
    st.markdown("""
    ¡Bienvenido a esta aplicación interactiva para explorar, entrenar y evaluar diversos modelos predictivos de Machine Learning!
    
    Esta herramienta está diseñada como una práctica para entender los conceptos de modelado en un contexto de Big Data, utilizando el famoso dataset **"California Housing"**. El objetivo es predecir el **valor mediano de una vivienda** en California basado en diferentes características.

    ### ¿Qué puedes hacer aquí?
    
    1.  **Navegar por las Secciones:** Utiliza el menú en el panel lateral para moverte entre las diferentes partes de la aplicación.
    2.  **Análisis Exploratorio de Datos (EDA):** Entiende la estructura y distribución de los datos con tablas y visualizaciones.
    3.  **Explorar Modelos Predictivos:**
        - Selecciona un modelo de la lista (Regresión Lineal, Árbol de Decisión, etc.).
        - Lee una breve explicación sobre cómo funciona cada modelo.
        - **Ajusta los hiperparámetros** del modelo usando los controles interactivos en el panel lateral.
        - Observa cómo cambian las métricas de rendimiento (Error Cuadrático Medio y R²) en tiempo real.
    4.  **Realizar Predicciones:** En la sección de cada modelo, puedes introducir tus propios valores para las características de una vivienda y obtener una predicción instantánea del precio.

    **¡Comienza explorando el dataset en la sección de "Análisis Exploratorio de Datos (EDA)" o salta directamente a entrenar un modelo!**
    """)

elif app_mode == "Análisis Exploratorio de Datos (EDA)":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.header("Dataset: California Housing")
    st.markdown("Este dataset contiene información de los grupos de bloques censales en California. Cada fila corresponde a un grupo de bloques.")
    
    st.subheader("Vistazo a los Datos")
    st.dataframe(df.head())
    
    st.subheader("Descripción de las Características")
    st.markdown("""
    - **MedInc:** Ingreso mediano en el grupo de bloques (en decenas de miles de dólares).
    - **HouseAge:** Antigüedad mediana de las viviendas en el grupo de bloques.
    - **AveRooms:** Número promedio de habitaciones por vivienda.
    - **AveBedrms:** Número promedio de dormitorios por vivienda.
    - **Population:** Población del grupo de bloques.
    - **AveOccup:** Ocupación promedio por vivienda (miembros por hogar).
    - **Latitude:** Latitud del centro del grupo de bloques.
    - **Longitude:** Longitud del centro del grupo de bloques.
    - **MedHouseVal (Target):** Valor mediano de la vivienda en el grupo de bloques (en cientos de miles de dólares).
    """)

    st.subheader("Descripción Estadística de los Datos")
    st.write(df.describe())

    st.subheader("Visualizaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribución del Valor Mediano de la Vivienda (Target)")
        st.markdown("El valor está en cientos de miles de dólares ($100,000s).")
        st.bar_chart(df['MedHouseVal'].value_counts().sort_index())
    
    with col2:
        st.markdown("#### Ubicación Geográfica de las Viviendas")
        st.markdown("El color representa el valor mediano de la vivienda (más oscuro = más caro).")
        
        map_df = df.copy()
        
        min_val = map_df['MedHouseVal'].min()
        max_val = map_df['MedHouseVal'].max()
        map_df['alpha'] = ((map_df['MedHouseVal'] - min_val) / (max_val - min_val)) * 255
        
        map_df['color'] = map_df['alpha'].apply(lambda alpha: [255, 90, 0, int(alpha)])
        
        map_df['size'] = map_df['Population'] * 0.01

        st.map(map_df, latitude='Latitude', longitude='Longitude', color='color', size='size')

else:
    model_name = app_mode
    st.title(f"🤖 Modelo Predictivo: {model_name}")

    model_descriptions = {
        "Regresión Lineal": "Un modelo simple que busca la mejor relación lineal entre las características y el valor de la vivienda. Es rápido y fácil de interpretar, pero puede no capturar relaciones complejas.",
        "Árbol de Decisión": "Este modelo aprende una serie de reglas de 'si-entonces' para dividir los datos y hacer una predicción. Es muy interpretable, pero propenso a sobreajustarse (memorizar) a los datos de entrenamiento.",
        "Random Forest": "Un modelo de ensamble que entrena muchos Árboles de Decisión sobre diferentes subconjuntos de datos y promedia sus predicciones. Es mucho más robusto y preciso que un solo árbol, pero menos interpretable.",
        "Gradient Boosting": "Otro modelo de ensamble que construye árboles de forma secuencial. Cada nuevo árbol intenta corregir los errores de los árboles anteriores. Suele ser uno de los modelos de mayor rendimiento.",
        "Support Vector Regressor (SVR)": "Busca encontrar un 'tubo' que contenga la mayor cantidad de puntos de datos posible, minimizando las desviaciones fuera de él. Es muy efectivo en espacios de alta dimensión y cuando las relaciones no son lineales, especialmente con el kernel 'rbf'."
    }
    st.markdown(model_descriptions.get(model_name, "Selecciona un modelo para ver su descripción."))
    st.divider()

    st.sidebar.header("Ajuste de Hiperparámetros")
    params = {}
    if model_name == 'Árbol de Decisión':
        params['max_depth'] = st.sidebar.slider("Profundidad Máxima (max_depth)", 1, 30, 10)
        params['min_samples_split'] = st.sidebar.slider("Mínimo de Muestras para Dividir (min_samples_split)", 2, 100, 2)
    elif model_name == 'Random Forest':
        params['n_estimators'] = st.sidebar.slider("Número de Árboles (n_estimators)", 10, 500, 100, step=10)
        params['max_depth'] = st.sidebar.slider("Profundidad Máxima (max_depth)", 1, 30, 10)
    elif model_name == 'Gradient Boosting':
        params['n_estimators'] = st.sidebar.slider("Número de Árboles (n_estimators)", 10, 500, 100, step=10)
        params['learning_rate'] = st.sidebar.slider("Tasa de Aprendizaje (learning_rate)", 0.01, 1.0, 0.1, step=0.01)
        params['max_depth'] = st.sidebar.slider("Profundidad Máxima (max_depth)", 1, 10, 3)
    elif model_name == 'Support Vector Regressor (SVR)':
        params['C'] = st.sidebar.slider("Parámetro de Regularización (C)", 0.01, 10.0, 1.0)
        params['kernel'] = st.sidebar.selectbox("Kernel", ['linear', 'rbf', 'poly'])
        params['gamma'] = st.sidebar.select_slider("Gamma", options=['scale', 'auto', 0.01, 0.1, 1])
    elif model_name == 'Red Neuronal (MLP)':
        params['learning_rate'] = st.sidebar.slider("Tasa de Aprendizaje (learning_rate)", 0.0001, 0.01, 0.001, format="%.4f")
        params['epochs'] = st.sidebar.slider("Épocas de Entrenamiento (epochs)", 10, 200, 50)
        st.sidebar.subheader("Arquitectura de la Red")
        params['neurons_layer1'] = st.sidebar.slider("Neuronas en Capa Oculta 1", 16, 128, 64, step=16)
        params['neurons_layer2'] = st.sidebar.slider("Neuronas en Capa Oculta 2", 8, 64, 32, step=8)
        
    X_train_to_use = X_train_scaled if model_name == 'Support Vector Regressor (SVR)' else X_train
    X_test_to_use = X_test_scaled if model_name == 'Support Vector Regressor (SVR)' else X_test

    with st.spinner("Entrenando el modelo con los parámetros seleccionados..."):
        model = train_model(model_name, X_train_to_use, y_train, params)
    st.success("¡Modelo entrenado exitosamente!")

    y_pred = model.predict(X_test_to_use)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.header("Evaluación del Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Error Cuadrático Medio (MSE)", f"{mse:.4f}")
        st.info("💡 **MSE:** Mide el promedio de los errores al cuadrado. Un valor más bajo es mejor.")
    with col2:
        st.metric("Coeficiente de Determinación (R²)", f"{r2:.4f}")
        st.info("💡 **R²:** Indica qué proporción de la varianza en el valor de la vivienda es predecible a partir de las características. Un valor más cercano a 1 es mejor.")

    st.divider()
    st.header("Realizar una Predicción en Vivo")
    st.markdown("Ajusta los siguientes valores para simular una nueva vivienda y predecir su precio.")
    
    input_data = {}
    prediction_cols = st.columns(2)
    
    for i, feature in enumerate(feature_names):
        with prediction_cols[i % 2]:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            input_data[feature] = st.slider(
                f"Valor para {feature}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float(std_val / 10)
            )

    if st.button("Predecir Valor de la Vivienda"):
        input_df = pd.DataFrame([input_data])
        
        if model_name == 'Support Vector Regressor (SVR)':
            input_df_scaled = scaler.transform(input_df)
            prediction = model.predict(input_df_scaled)
        else:
            prediction = model.predict(input_df)
        
        st.subheader("Resultado de la Predicción")
        st.success(f"El valor mediano estimado para la vivienda es: **${prediction[0] * 100000:,.2f}**")

    with st.expander("Ver el código de Python para este modelo"):
        code = f"""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.{'linear_model' if model_name == 'Regresión Lineal' else 'tree' if model_name == 'Árbol de Decisión' else 'ensemble' if model_name in ['Random Forest', 'Gradient Boosting'] else 'svm'}.{model.__class__.__name__}
        from sklearn.metrics import mean_squared_error, r2_score

        # Nota: X_train, y_train, X_test, y_test son los datos pre-divididos.
        # Para SVR, se usarían X_train_scaled y X_test_scaled.

        # 1. Inicializar el modelo con los parámetros seleccionados
        params = {params}
        model = {model.__class__.__name__}(**params, random_state=42) if 'random_state' in {model.__class__}().get_params() else {model.__class__.__name__}(**params)

        # 2. Entrenar el modelo
        model.fit(X_train, y_train)

        # 3. Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)

        # 4. Evaluar el rendimiento
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {{mse:.4f}}")
        print(f"R²: {{r2:.4f}}")
        """
        st.code(code, language='python')
