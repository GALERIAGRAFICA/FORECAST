import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

# LOGO GG
# Cargar el logo
logo = Image.open("Logo.png")

# Mostrar el logo en la app
st.image(logo, width=150, use_column_width=True)

# Título de la app
st.title('Predicción de Consumo de Papel')

# Subir el archivo Excel
uploaded_file = st.file_uploader("Sube el archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    # Cargar el archivo Excel
    df = pd.read_excel(uploaded_file, sheet_name='00GG_CONTROL_CONSUMOS')

    # Convertir 'Fecha' a formato datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    # Extraer el año y mes, asegurando que los valores sean enteros
    df['Año'] = pd.to_numeric(df['Fecha'].dt.year, errors='coerce', downcast='integer')
    df['Mes'] = pd.to_numeric(df['Fecha'].dt.month, errors='coerce', downcast='integer')

    # Verificar si hay valores nulos en las columnas 'Año' y 'Mes'
    if df[['Año', 'Mes']].isnull().values.any():
        st.error("Los valores de 'Año' o 'Mes' no son válidos. Por favor revisa los datos.")
    else:
        # Crear una columna 'Fecha' con el día asignado manualmente como 1
        df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01', errors='coerce')

        # Verificar si la creación de la columna 'Fecha' resultó en valores nulos
        if df['Fecha'].isnull().values.any():
            st.error("Error al crear la columna de fecha. Verifica los valores de 'Año' y 'Mes'.")
        else:
            # Agrupar los datos por las columnas necesarias
            grouped_consumption = df.groupby(['Papeles.Familia', 'Papeles.SubFamilia', 'CERTIFICADO', 'Papeles.Gramaje', 'MARCA', 'Año', 'Mes']).agg({
                'KG Consumidos': 'sum'
            }).reset_index()

            # Filtrado según subfamilia y gramaje para el dataframe de consumo
            subfamilias = grouped_consumption['Papeles.SubFamilia'].unique()
            subfamilia_seleccionada = st.selectbox("Selecciona la subfamilia", subfamilias)

            subfamilia_filtered = grouped_consumption[grouped_consumption['Papeles.SubFamilia'] == subfamilia_seleccionada]

            gramajes_unicos = subfamilia_filtered['Papeles.Gramaje'].unique()
            gramaje_seleccionado = st.selectbox("Selecciona el gramaje (opcional)", ['Ninguno'] + list(gramajes_unicos))

            if gramaje_seleccionado != 'Ninguno':
                gramaje_filtered = subfamilia_filtered[subfamilia_filtered['Papeles.Gramaje'] == gramaje_seleccionado]
            else:
                gramaje_filtered = subfamilia_filtered

            # Tercer nivel de filtrado: Seleccionar si se quiere filtrar por marca o certificado
            tipo_analisis = st.selectbox("Selecciona análisis adicional (opcional)", ["Ninguno", "Marca", "Certificado"])

            if tipo_analisis == "Marca":
                # Filtrar por marca
                marcas_unicas = gramaje_filtered['MARCA'].unique()
                marca_seleccionada = st.selectbox("Selecciona la marca", marcas_unicas)

                # Filtrar por la marca seleccionada
                analisis_filtered = gramaje_filtered[gramaje_filtered['MARCA'] == marca_seleccionada]

            elif tipo_analisis == "Certificado":
                # Filtrar por certificado
                certificados_unicos = gramaje_filtered['CERTIFICADO'].unique()
                certificado_seleccionado = st.selectbox("Selecciona el certificado", certificados_unicos)

                # Filtrar por el certificado seleccionado
                analisis_filtered = gramaje_filtered[gramaje_filtered['CERTIFICADO'] == certificado_seleccionado]

            else:
                # Si no se selecciona análisis adicional, usamos el filtrado hasta el nivel anterior
                analisis_filtered = gramaje_filtered

            # Agrupar los datos por mes y año para la predicción
            prophet_data = analisis_filtered.groupby(['Año', 'Mes']).agg({'KG Consumidos': 'sum'}).reset_index()

            # Crear una nueva columna 'Fecha' para combinar año y mes
            prophet_data['Fecha'] = pd.to_datetime(prophet_data['Año'].astype(str) + '-' + prophet_data['Mes'].astype(str) + '-01', errors='coerce')

            # Predicción usando Prophet
            df_prophet = prophet_data[['Fecha', 'KG Consumidos']].rename(columns={'Fecha': 'ds', 'KG Consumidos': 'y'})

            # Verificar si hay suficientes datos para Prophet
            if len(df_prophet) >= 2:
                model_prophet = Prophet()
                model_prophet.fit(df_prophet)

                future = model_prophet.make_future_dataframe(periods=12, freq='M')
                forecast = model_prophet.predict(future)

                # Graficar los resultados
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_prophet['ds'], df_prophet['y'], label='Datos Reales', color='blue')
                ax.plot(forecast['ds'], forecast['yhat'], label='Predicción', color='red')
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)

                # Mostrar el gráfico en la app de Streamlit
                plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}')
                if gramaje_seleccionado != 'Ninguno':
                    plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}, Gramaje: {gramaje_seleccionado}')
                if tipo_analisis == "Marca":
                    plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}, Marca: {marca_seleccionada}')
                elif tipo_analisis == "Certificado":
                    plt.title(f'Predicción de Consumo (KG) - Subfamilia: {subfamilia_seleccionada}, Certificado: {certificado_seleccionado}')

                plt.xlabel('Fecha')
                plt.ylabel('KG Consumidos')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
                
                 # Agrupar los datos por mes y año
                df_mensual = analisis_filtered.groupby(['Año', 'Mes'])['KG Consumidos'].sum().reset_index()

                # Calcular las estadísticas descriptivas globales para el consumo mensual
                consumo_mensual_mean = df_mensual['KG Consumidos'].mean()
                consumo_mensual_std = df_mensual['KG Consumidos'].std()
                consumo_mensual_min = df_mensual['KG Consumidos'].min()
                consumo_mensual_max = df_mensual['KG Consumidos'].max()

                # Mostrar las estadísticas descriptivas globales
                st.write(f"**Estadísticas descriptivas globales del consumo mensual para {subfamilia_seleccionada}:**")
                st.write(f"- Media mensual de consumo: {consumo_mensual_mean:.2f} KG")
                st.write(f"- Desviación estándar mensual: {consumo_mensual_std:.2f} KG")
                st.write(f"- Consumo mínimo mensual: {consumo_mensual_min:.2f} KG")
                st.write(f"- Consumo máximo mensual: {consumo_mensual_max:.2f} KG")

            else:
                st.write(f"No hay suficientes datos para realizar la predicción para la subfamilia {subfamilia_seleccionada}.")

            # -------------------------------------------------------------------
            # GEMELO 2: PARA EL ANÁLISIS DE CLUSTERING (con la columna 'Papeles.Dim.1 (mm)')
            # Aquí no agregamos, simplemente seleccionamos las columnas de interés incluyendo 'Papeles.Dim.1 (mm)'

            clustering_data = df[['Papeles.Familia', 'Papeles.SubFamilia', 'CERTIFICADO', 'Papeles.Gramaje', 'MARCA', 'Año', 'Mes', 'Papeles.Dim.1 (mm)', 'KG Consumidos']]

            # Aplicar el mismo filtro de subfamilia y gramaje que aplicamos antes
            clustering_filtered = clustering_data[clustering_data['Papeles.SubFamilia'] == subfamilia_seleccionada]

            if gramaje_seleccionado != 'Ninguno':
                clustering_filtered = clustering_filtered[clustering_filtered['Papeles.Gramaje'] == gramaje_seleccionado]

            if tipo_analisis == "Marca":
                analisis_filtered1 = clustering_filtered[clustering_filtered['MARCA'] == marca_seleccionada]

            elif tipo_analisis == "Certificado":
                analisis_filtered1 = clustering_filtered[clustering_filtered['CERTIFICADO'] == certificado_seleccionado]

            else:
                # Si no se selecciona análisis adicional, usamos el filtrado hasta el nivel anterior
                analisis_filtered1 = clustering_filtered

            # Preparar los datos para clustering usando 'Papeles.Dim.1 (mm)'
            widths = analisis_filtered1['Papeles.Dim.1 (mm)'].values.reshape(-1, 1)

            # Paso 1: Análisis del método del codo
            max_clusters = st.slider('Selecciona el número máximo de clusters para el análisis del codo', min_value=2, max_value=20, value=10)

            wcss = []
            for i in range(1, max_clusters + 1):
                kmeans_temp = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans_temp.fit(widths)
                wcss.append(kmeans_temp.inertia_)

            # Graficar el análisis del codo
            fig_codo, ax_codo = plt.subplots()
            ax_codo.plot(range(1, max_clusters + 1), wcss)
            ax_codo.set_title('Método del Codo para k Óptimo')
            ax_codo.set_xlabel('Número de clusters')
            ax_codo.set_ylabel('WCSS')
            ax_codo.grid(True)
            st.pyplot(fig_codo)

            # Paso 2: Aplicar el clustering
            n_clusters = st.slider('Selecciona el número de clusters para aplicar K-Means', min_value=2, max_value=10, value=4)

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            cluster_labels = kmeans.fit_predict(widths)

            # Agregar las etiquetas de cluster al dataframe
            analisis_filtered1['Cluster'] = cluster_labels

            # --------------------------------------------------------------
            # Calcular la demanda total para cada cluster
            cluster_demand = analisis_filtered1.groupby('Cluster')['KG Consumidos'].sum().reset_index()
            cluster_demand['Centroid Width (mm)'] = kmeans.cluster_centers_

            # Calcular el mínimo y máximo de los anchos para cada cluster
            cluster_min_max = analisis_filtered1.groupby('Cluster')['Papeles.Dim.1 (mm)'].agg(['min', 'max']).reset_index()

            # Ordenar por el ancho mínimo de cada cluster
            cluster_min_max_sorted = cluster_min_max.sort_values('min')

            # Visualizar el resultado ordenado
            st.write("Demanda total por cluster y sus centroides:")
            st.dataframe(cluster_demand)

            st.write("Rango de anchos para cada cluster (ordenado por el ancho mínimo):")
            st.dataframe(cluster_min_max_sorted)

            # --------------------------------------------------------------
            # Visualizar los clusters en el tiempo y su consumo real
            # Agregar de nuevo la columna 'Fecha' al dataframe si fue eliminada en el proceso de filtrado/agrupación
            analisis_filtered1['Fecha'] = pd.to_datetime(analisis_filtered1['Año'].astype(str) + '-' + analisis_filtered1['Mes'].astype(str) + '-01', errors='coerce')
           # Ahora que 'Fecha' está disponible, puedes continuar con el gráfico
           
           
           # Agrupar por 'Fecha' y 'Cluster' para obtener el consumo total por cluster en cada fecha
            consumo_agrupado1 = analisis_filtered1.groupby(['Fecha', 'Cluster'])['KG Consumidos'].sum().reset_index()


            # Visualizar los clusters en el tiempo con un gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6)) 
            # Ajustar el ancho de las barras
            bar_width = 1.2  # Cambia este valor para hacer las barras más gruesas


            # Crear un gráfico de barras para cada cluster
            for cluster in consumo_agrupado1['Cluster'].unique():
                cluster_data = consumo_agrupado1[consumo_agrupado1['Cluster'] == cluster]
                ax.bar(cluster_data['Fecha'], cluster_data['KG Consumidos'],width=bar_width, label=f'Cluster {cluster}')

            ax.set_title('Consumo Real por Cluster en el Tiempo')
            ax.set_xlabel('Fecha')
            ax.set_ylabel('KG Consumidos')
            ax.legend()
            ax.grid(True)

            # Rotar las etiquetas del eje x para mayor claridad
            plt.xticks(rotation=45)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            
            import streamlit as st

            # Identificar los anchos más comunes
            common_widths = analisis_filtered1['Papeles.Dim.1 (mm)'].value_counts().head(10)
            st.write("Anchos más comunes:")
            st.dataframe(common_widths)

            # Entrada de datos de usuario para los anchos de bobinas
            anchos_bobinas_input = st.text_input('Introduce los anchos de bobinas separados por comas (ej: 2100, 1860)', '2100, 1860')

            # Convertir la entrada del usuario en una lista de enteros
            anchos_bobinas = [int(ancho.strip()) for ancho in anchos_bobinas_input.split(',')]

            # Función para evaluar qué pliegos caben en cada bobina
            def evaluar_bobinas(anchos_bobinas, anchos_pliegos):
                resultados = {}
                for bobina in anchos_bobinas:
                    combinaciones = []
                    for ancho in anchos_pliegos:
                        cantidad = bobina // ancho
                        if cantidad > 0:
                            combinaciones.append((ancho, cantidad))
                    resultados[bobina] = combinaciones
                return resultados

            # Ejemplo de anchos de pliegos agrupados por cluster
            anchos_minimos = cluster_min_max_sorted['min'].tolist()
            anchos_maximos = cluster_min_max_sorted['max'].tolist()

            # Evaluar qué pliegos caben en cada bobina propuesta (MAX)
            resultados_max = evaluar_bobinas(anchos_bobinas, anchos_maximos)
            st.write("Resultados (MAX):")
            for bobina, combinaciones in resultados_max.items():
                st.write(f"Bobina de {bobina} mm puede acomodar (MAX):")
                for combinacion in combinaciones:
                    st.write(f"  - {combinacion[1]} pliegos de {combinacion[0]} mm")

            # Evaluar qué pliegos caben en cada bobina propuesta (MIN)
            resultados_min = evaluar_bobinas(anchos_bobinas, anchos_minimos)
            st.write("Resultados (MIN):")
            for bobina, combinaciones in resultados_min.items():
                st.write(f"Bobina de {bobina} mm puede acomodar (MIN):")
                for combinacion in combinaciones:
                    st.write(f"  - {combinacion[1]} pliegos de {combinacion[0]} mm")

