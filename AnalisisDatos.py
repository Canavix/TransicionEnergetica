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
dfPromedio = dfAgg.groupby('Country')[['Electricity Generation (GWh)', 'Electricity Installed Capacity (MW)']].mean().reset_index()
#"""
# Mostrar promedio de Generación y capacidad instalada
print(dfPromedio.head())
# Copiar al portapapeles los datos del promedio (para verificación)
dfPromedio.to_clipboard(decimal=",")
#"""
# Analisis de datos .......................................................................................................

"""
# Calcular la generación promedio y la capacidad para cada país en los últimos 10 años
df_avg = df_agg.groupby('Country')[['Electricity Generation (GWh)', 'Electricity Installed Capacity (MW)']].mean().reset_index()

# Seleccionar los 3 países principales por generación promedio
top_3_countries = df_avg.nlargest(3, 'Electricity Generation (GWh)')

# Imprimir los 3 países principales
print("Los 3 países principales en generación de energía solar fotovoltaica en América Latina en los últimos 10 años son:")
print(top_3_countries[['Country', 'Electricity Generation (GWh)']].to_markdown(index=False, numalign="left", stralign="left"))

# Filtrar datos para los 3 países principales y Colombia
top_countries_and_colombia = df_avg[df_avg['Country'].isin(top_3_countries['Country'].tolist() + ['Colombia'])]

# Crear una tabla dinámica con 'Year' como índice, 'Country' como columnas y 'Electricity Installed Capacity (MW)' como valores
pivot_table = df_agg.pivot(index='Year', columns='Country', values='Electricity Installed Capacity (MW)')

# Imprimir la tabla dinámica
print("\nTabla dinámica de capacidad instalada para los 3 países principales y Colombia:")
print(pivot_table.to_markdown(numalign="left", stralign="left"))

# Crear un gráfico de líneas a partir de la tabla dinámica
chart = alt.Chart(pivot_table.reset_index().melt('Year', var_name='País', value_name='Capacidad (MW)')).mark_line().encode(
    x='Year',
    y='Capacidad (MW)',
    color='País',
    tooltip=['Year', 'País', 'Capacidad (MW)']
).properties(
    title='Evolución de la capacidad instalada de energía solar fotovoltaica (MW)'
).interactive()

# Mostrar el gráfico
chart.save('evolucion_capacidad_instalada.json')

# Calcular la correlación entre 'Electricity Installed Capacity (MW)' y 'Electricity Generation (GWh)'
correlation = df_agg['Electricity Installed Capacity (MW)'].corr(df_agg['Electricity Generation (GWh)'])

# Mostrar el resultado
print(f"\nLa correlación entre la capacidad instalada y la generación de energía solar fotovoltaica es: {correlation:.2f}")
"""