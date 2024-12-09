import pandas as pd
import altair as alt

# Leer el archivo CSV
df = pd.read_csv('Datos/Dataset.csv', sep=";", decimal=",")
"""
# Datos del dataset original ..............................................................................................
# Verificar valores nulos en el dataset
print("\n# Verificar valores nulos en el dataset.......................................")
print(df.isnull().sum())
# Información general del dataset
print("\n# Información general del dataset.............................................")
print(df.info())
# Descripción estadística del dataset
print("\n# Descripción estadística del dataset.........................................")
print(df.describe())
# Mostrar estructura del dataset original
print("\nEstructura original del dataset ................................................")
print(df.head())
# Datos del dataset original ..............................................................................................
"""

# Limpieza de datos .......................................................................................................
#Convetrir la columna de Year a numérica de tipo entero, convirtiendo
df['Year'] = pd.to_numeric(df['Year'], downcast="integer", errors='coerce')
# Eliminar filas con valores faltantes en la columna 'Year'
df.dropna(subset=['Year'], inplace=True)

# Convertir la columna a tipo texto (string) antes de aplicar .strip()
df['Electricity Generation (GWh)'] = df['Electricity Generation (GWh)'].astype(str).str.strip()
# Eliminar espacios en blanco al principio y al final de la columna de Generación eléctrica
df['Electricity Generation (GWh)'] = df['Electricity Generation (GWh)'].str.strip()
# Reemplazar las comas por puntos en la columna 'Electricity Generation (GWh)'
df['Electricity Generation (GWh)'] = df['Electricity Generation (GWh)'].astype(str).str.replace(',', '.')
# Convertir la columna de Generación eléctrica a numérica, convirtiendo valores no numéricos a NaN
df['Electricity Generation (GWh)'] = pd.to_numeric(df['Electricity Generation (GWh)'], errors='coerce')
# Redondear los valores de la columna de Generación eléctrica a dos cifras decimales
df['Electricity Generation (GWh)'] = df['Electricity Generation (GWh)'].apply(lambda x: round(x, 2))
# Eliminar filas con valores NaN en la columna de Generación eléctrica
df.dropna(subset=['Electricity Generation (GWh)'], inplace=True)

# Convertir la columna a tipo texto (string) antes de aplicar .strip()
df['Electricity Installed Capacity (MW)'] = df['Electricity Installed Capacity (MW)'].astype(str).str.strip()
# Eliminar espacios en blanco al principio y al final de la columna de Capacidad eléctrica instalada
df['Electricity Installed Capacity (MW)'] = df['Electricity Installed Capacity (MW)'].str.strip()
# Reemplazar las comas por puntos en la columna 'Electricity Installed Capacity (MW)'
df['Electricity Installed Capacity (MW)'] = df['Electricity Installed Capacity (MW)'].astype(str).str.replace(',', '.')
# Convertir la columna de Capacidad eléctrica instalada a numérica, convirtiendo valores no numéricos a NaN
df['Electricity Installed Capacity (MW)'] = pd.to_numeric(df['Electricity Installed Capacity (MW)'], errors='coerce')
# Redondear los valores de la columna de Capacidad eléctrica instalada a dos cifras decimales
df['Electricity Installed Capacity (MW)'] = df['Electricity Installed Capacity (MW)'].apply(lambda x: round(x, 2))
# Eliminar filas con valores NaN en la columna de Capacidad eléctrica instalada
df.dropna(subset=['Electricity Installed Capacity (MW)'], inplace=True)
# Limpieza de datos .......................................................................................................

# Filtrado de datos .......................................................................................................
# Filtrar datos por paises
dfFiltrado = df[(df['Country'].str.contains("Colombia")) | 
                (df['Country'].str.contains("Argentina")) |
                (df['Country'].str.contains("Brazil")) |
                (df['Country'].str.contains("Chile")) |
                (df['Country'].str.contains("Ecuador")) |
                (df['Country'].str.contains("Mexico")) |
                (df['Country'].str.contains("Paraguay")) |
                (df['Country'].str.contains("Peru")) |
                (df['Country'].str.contains("Uruguay"))
                ]

# Filtrar datos por Energía Solar Fotovoltaica
dfFiltrado = dfFiltrado[(dfFiltrado['Group Technology'].str.contains("Solar energy")) &
                        (dfFiltrado['Technology'].str.contains("Solar photovoltaic")) &
                        (dfFiltrado['Sub-Technology'].str.contains("On-grid Solar photovoltaic"))
                        ]

# Filtrar datos de los últimos 6 años
dfFiltrado = dfFiltrado[dfFiltrado['Year'] >= dfFiltrado['Year'].max() - 5]
"""
# Información general del dataset filtrado
print("\nInformación del dataset filtrado ..................................................")
print(dfFiltrado.info())
# Mostrar dataset filtrado
print(dfFiltrado.head())
# Copia al portapapeles el dataset filtrado (para verificación)
dfFiltrado.to_clipboard(decimal=",")
"""

#Agrupar datos por país y año, y calcular la suma de Generación Eléctrica (GWh) y de Capacidad Eléctrica Instalada (MW)
dfAgg = dfFiltrado.groupby(['Country', 'Year'])[['Electricity Generation (GWh)', 'Electricity Installed Capacity (MW)']].sum().reset_index()
"""
# Mostrar datos agrupados por pais y año
print("\nDatos agrupados por pais y año....................................................")
print(dfAgg.head())
# Copiar al portapapeles los datos agrupados (para verificación)
dfAgg.to_clipboard(decimal=",")
"""
# Filtrado de datos .......................................................................................................

# Analisis de datos .......................................................................................................
# Calcular la generación promedio y la capacidad para cada país en los últimos 6 años
dfPromedio = dfAgg.groupby('Country',)[['Electricity Generation (GWh)', 'Electricity Installed Capacity (MW)']].mean().reset_index()
"""
# Mostrar promedio de Generación y capacidad instalada
print(dfPromedio.head())
# Copiar al portapapeles los datos del promedio (para verificación)
dfPromedio.to_clipboard(decimal=",")
"""

# Los 3 países con mayor promedio de Generación eléctrica
top3Generacion = dfPromedio.nlargest(3, 'Electricity Generation (GWh)')
# Mostrar los 3 países
print("\nLos 3 países con mayor generación de energía solar fotovoltaica en América Latina en los últimos 6 años son:\n")
print(top3Generacion[['Country', 'Electricity Generation (GWh)']].to_markdown(index=False, numalign="center", stralign="left"))
print("\n")

# Los 3 países con mayor promedio de Capacidad eléctrica instalada
top3Capacidad = dfPromedio.nlargest(3, 'Electricity Installed Capacity (MW)')
# Mostrar los 3 países
print("\nLos 3 países con mayor incremento de capacidad instalada para energía solar fotovoltaica en América Latina en los últimos 6 años son:\n")
print(top3Capacidad[['Country', 'Electricity Installed Capacity (MW)']].to_markdown(index=False, numalign="center", stralign="left"))
print("\n")

# Los 3 paises principales y Colombia - Generación
topColombiaGeneracion = dfPromedio[dfPromedio['Country'].isin(top3Generacion['Country'].tolist() + ['Colombia'])]
# Mostrar los 3 países y Colombia
print("\nGeneración de energía solar fotovoltaica en Amética Latina en los últimos 6 años:\n")
print(topColombiaGeneracion[['Country', 'Electricity Generation (GWh)']].to_markdown(index=False, numalign="center", stralign="left"))
print("\n")

# Los 3 países principales y Colombia - Capacidad
topColombiaCapacidad = dfPromedio[dfPromedio['Country'].isin(top3Capacidad['Country'].tolist() + ['Colombia'])]
# Mostrar los 3 países y Colombia
print("\nCapacidad instalada para energía solar fotovoltaica en América Latina en los últimos 6 años:\n")
print(topColombiaCapacidad[['Country', 'Electricity Installed Capacity (MW)']].to_markdown(index=False, numalign="center", stralign="left"))
print("\n")
# Analisis de datos .......................................................................................................

# Tabla dinámica con los datos ............................................................................................
# Tabla dinámica con 'Year' como índice, 'Country' como columnas y 'Electricity Generatio (GWh)' como valores
# Define la variable paisesPrincipales con los nombres de los 3 países con mayor generación y Colombia
paisesPrincipalesGeneracion = topColombiaGeneracion['Country'].tolist()
# Filtra el DataFrame dfAgg para los paisesPrincipales y Colombia
dfPrincipalesGeneracion = dfAgg[dfAgg['Country'].isin(paisesPrincipalesGeneracion)]

# Crea la tabla dinámica con el DataFrame filtrado
tablaDinamicaGeneracion = dfPrincipalesGeneracion.pivot(index='Year', columns='Country', values='Electricity Generation (GWh)')
# Imprime la tabla dinámica
print("\nTabla dinámica de generación eléctrica para los 3 países principales y Colombia:")
print(tablaDinamicaGeneracion.to_markdown(numalign="left", stralign="left"))
print("\n")


# Tabla dinámica con 'Year' como índice, 'Country' como columnas y 'Electricity Installed Capacity (MW)' como valores
# Define la variable paisesPrincipales con los nombres de los 3 países con mayor capacidad instalada y Colombia
paisesPrincipalesCapacidad = topColombiaCapacidad['Country'].tolist()
# Filtra el DataFrame dfAgg para los paisesPrincipales y Colombia
dfPrincipalesCapacidad = dfAgg[dfAgg['Country'].isin(paisesPrincipalesCapacidad)]

# Crea la tabla dinámica con el DataFrame filtrado
tablaDinamicaCapacidad = dfPrincipalesCapacidad.pivot(index='Year', columns='Country', values='Electricity Installed Capacity (MW)')
# Imprime la tabla dinámica
print("\nTabla dinámica de capacidad instalada para los 3 países principales y Colombia:")
print(tablaDinamicaCapacidad.to_markdown(numalign="left", stralign="left"))
print("\n")
# Tabla dinámica con los datos ............................................................................................


# Gráficos ................................................................................................................
# Convertir la tabla dinámica a un formato adecuado para Altair
tablaGeneracion = tablaDinamicaGeneracion.reset_index().melt('Year', var_name='País', value_name='Electricidad Generada (GWh)')
# Crear el gráfico de barras agrupadas
graficoGeneracion = alt.Chart(tablaGeneracion).mark_bar().encode(
    x=alt.X('País', axis=None),
    y='Electricidad Generada (GWh)',
    color='País',
    column=alt.Column('Year', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
    tooltip=['Year', 'País', 'Electricidad Generada (GWh)']
).properties(
    width=200,
    height=600,
    title='Electricidad Generada por País y Año'
)
# Crear archivo del gráfico
graficoGeneracion.save('Gráficos/ElectricidadGenerada.html')

# Convertir la tabla dinámica a un formato adecuado para Altair
tablaCapacidad = tablaDinamicaCapacidad.reset_index().melt('Year', var_name='País', value_name='Capacidad Instalada (MW)')
# Crear el gráfico de barras agrupadas
graficoCapacidad = alt.Chart(tablaCapacidad).mark_bar().encode(
    x=alt.X('País', axis=None),
    y='Capacidad Instalada (MW)',
    color='País',
    column=alt.Column('Year', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
    tooltip=['Year', 'País', 'Capacidad Instalada (MW)']
).properties(
    width=200,
    height=600,
    title='Capacidad Instalada por País y Año'
)
# Crear archivo del gráfico
graficoCapacidad.save('Gráficos/CapacidadInstalada.html')

# Mapa de dispersión con la media de generación
graficoDispersion = alt.Chart(dfPromedio).mark_circle(size=200).encode(
    x='Electricity Generation (GWh)',
    y='Electricity Installed Capacity (MW)',
    color='Country',
    tooltip=['Country', 'Electricity Generation (GWh)', 'Electricity Installed Capacity (MW)']
).properties(
    width=800,
    height=600,
    title='Relación entre capacidad instalada y generación de energía'
).interactive()
#Crear archivo del gráfico
graficoDispersion.save('Gráficos/RelacionGeneracionCapacidadDispersion.html')
# Gráficos ................................................................................................................

# Mapa coroplético ........................................................................................................
# Cargo el archivo de GeoJSON
geojsonData = alt.Data(url='Mapas/NASA.geojson', format=alt.DataFormat(property='features', type='json'))

# Crear el mapa
mapa = alt.Chart(geojsonData).mark_geoshape(
    fill='white',
    stroke='black',
    strokeWidth=1.5
).encode(
    color='Electricity Generation (GWh):Q'
).transform_lookup(
    lookup='admin',
    from_=alt.LookupData(dfPromedio, 'Country', ['Electricity Generation (GWh)'])
).properties(
    width=800,
    height=600,
    title='Mapa de Coropletas'
).project(
    type='mercator'
)

mapa.save('Gráficos/MapaCoropletasAmerica.html')
# Mapa coroplético ........................................................................................................

# Otros cálculos ..........................................................................................................
# Calcular la correlación entre 'Electricity Installed Capacity (MW)' y 'Electricity Generation (GWh)'
correlacion = dfAgg['Electricity Installed Capacity (MW)'].corr(dfAgg['Electricity Generation (GWh)'])
# Mostrar el resultado
print(f"\nLa correlación entre la capacidad instalada y la generación de energía solar fotovoltaica es: {correlacion:.2f}\n")
# Otros cálculos ..........................................................................................................