import pandas as pd
import altair as alt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Leer el archivo CSV
df = pd.read_csv('Datos/Dataset.csv', sep=";", decimal=",")

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
                #(df['Country'].str.contains("China")) |
                #(df['ISO3 code'].str.contains("USA")) |
                (df['Country'].str.contains("Uruguay"))
                ]

# Filtrar datos por Energía Solar Fotovoltaica
dfFiltrado = dfFiltrado[(dfFiltrado['Group Technology'].str.contains("Solar energy")) &
                        (dfFiltrado['Technology'].str.contains("Solar photovoltaic")) &
                        (dfFiltrado['Sub-Technology'].str.contains("On-grid Solar photovoltaic"))
                        ]

# Filtrar datos de los últimos 6 años
dfFiltrado = dfFiltrado[dfFiltrado['Year'] >= dfFiltrado['Year'].max() - 9]

#Agrupar datos por país y año, y calcular la suma de Generación Eléctrica (GWh) y de Capacidad Eléctrica Instalada (MW)
dfAgg = dfFiltrado.groupby(['Country', 'Year'])[['Electricity Generation (GWh)', 'Electricity Installed Capacity (MW)']].sum().reset_index()
# Filtrado de datos .......................................................................................................

# Analisis de datos .......................................................................................................
# Calcular la generación promedio y la capacidad para cada país en los últimos 6 años
dfPromedio = dfAgg.groupby('Country',)[['Electricity Generation (GWh)', 'Electricity Installed Capacity (MW)']].mean().reset_index()

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

# Interpolación ...........................................................................................................
# Se filtran los datos para trabajar únicamente con Colombia
datosColombiaGeneracion = dfPrincipalesGeneracion[dfPrincipalesGeneracion['Country'].str.contains("Colombia")]
print(datosColombiaGeneracion.to_markdown(numalign='center', stralign='left'))
print("\n")
#datosColombiaGeneracion.to_clipboard(decimal=',')

xGeneracion = datosColombiaGeneracion['Year']
yGeneracion = datosColombiaGeneracion['Electricity Generation (GWh)']
# Crear una función de interpolación (puedes elegir el tipo que necesites: 'linear', 'cubic', etc.)
f = interp1d(xGeneracion, yGeneracion, kind='cubic')
# Generar nuevos puntos para la interpolación
x_nuevo = np.linspace(2017, 2022, num=100, endpoint=True)
y_nuevo = f(x_nuevo)
# Graficar los datos originales y la interpolación
plt.plot(xGeneracion, yGeneracion, 'o', label='Datos originales')
plt.plot(x_nuevo, y_nuevo, '-', label='Interpolación')
plt.legend()
#plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generación Energética Solar en Colombia')
plt.show()

datosColombiaCapacidad = dfPrincipalesCapacidad[dfPrincipalesCapacidad['Country'].str.contains("Colombia")]
xCapacidad = datosColombiaCapacidad['Year']
yCapacidad = datosColombiaCapacidad['Electricity Installed Capacity (MW)']
# Crear una función de interpolación (puedes elegir el tipo que necesites: 'linear', 'cubic', etc.)
f = interp1d(xCapacidad, yCapacidad, kind='cubic')
# Generar nuevos puntos para la interpolación
x_nuevo = np.linspace(2017, 2022, num=100, endpoint=True)
y_nuevo = f(x_nuevo)
# Graficar los datos originales y la interpolación
plt.plot(xCapacidad, yCapacidad, 'o', label='Datos originales')
plt.plot(x_nuevo, y_nuevo, '-', label='Interpolación')
plt.legend()
#plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Capacidad Instalada - Solar en Colombia')
plt.show()
# Interpolación ...........................................................................................................

# Regresión Manual ........................................................................................................
"""
plt.scatter(xGeneracion,yGeneracion)
plt.xlabel("Electricidad Generada (GWh)")
plt.ylabel("Año")
plt.title("Generación Eléctrica - Solar Fotovoltaica en Colombia")
plt.grid()
plt.show()

matriz = datosColombiaGeneracion[['Year', 'Electricity Generation (GWh)']].to_numpy()
print(matriz)

# Hallamos los valores necesarios para la regresión
# Número de datos 
n = len(matriz)
# La sumatoria de Xi
sumatoriaX = np.sum(matriz[:,0])
# La sumatoria de Yi
sumatoriaY = np.sum(matriz[:,1])
# La sumatoria de Xi * Yi
sumatoriaXY = np.sum(matriz[:,0] * matriz[:,1])
# La sumatoria de X^2
sumatoriaX2 = np.sum(matriz[:,0] ** 2)

print("\nn: ", n)
print("SumX: ", sumatoriaX)
print("SumY: ", sumatoriaY)
print("SumXY: ", sumatoriaXY)
print("SumX^2: ", sumatoriaX2)

b1 = ((n * sumatoriaXY) - (sumatoriaX * sumatoriaY)) / ((n * sumatoriaX2) - (sumatoriaX) ** 2)
b0 = (sumatoriaY - (b1 * sumatoriaX)) / (n)

print("\nb1: ", b1)
print("b0: ", b0)
"""
# Regresión Manual ........................................................................................................

# Regresión con sklearn ...................................................................................................
varX = 'Year'
varY = 'Electricity Generation (GWh)'
varZ = 'Electricity Installed Capacity (MW)'

# Creamos un modelo para la regresión lineal
modeloGeneracion = LinearRegression()
# Le cargamos al modelo los datos de la variable independiente y la dependiente
modeloGeneracion.fit(datosColombiaGeneracion[[varX]], datosColombiaGeneracion[varY])

# Obtenemos la ecuación de la recta
print("\nEcuación de la recta: y = ", round(modeloGeneracion.coef_[0],3), "x + ", round(modeloGeneracion.intercept_,3))
# Obtenemos el coeficiente de correlación
print("Coeficiente de correlación: ", round(np.corrcoef(datosColombiaGeneracion[varX], datosColombiaGeneracion[varY])[0,1],3))
# Obtenemos el coeficiente de determinación
print("Coeficiente de determinación: ", round(r2_score(datosColombiaGeneracion[varY], modeloGeneracion.predict(datosColombiaGeneracion[[varX]])),3))

datoPrediccion = 2024
print("\nPredicción de generación en el año ", datoPrediccion, ": ", modeloGeneracion.predict([[datoPrediccion]]))

# Graficar 
plt.plot(datosColombiaGeneracion[varX], datosColombiaGeneracion[varY], label='y')
plt.plot(datosColombiaGeneracion[varX], modeloGeneracion.predict(datosColombiaGeneracion[[varX]]), label='predicción')
plt.title("Regresión lineal - Generación Eléctrica Colombia")
plt.xlabel("Energía generada (GWh)")
plt.ylabel("Año")
plt.legend()
plt.grid()
plt.show()

# Creamos un modelo para la regresión lineal para la capacidad
modeloCapacidad = LinearRegression()
# Le cargamos al modelo los datos de la variable independiente y la dependiente
modeloCapacidad.fit(datosColombiaCapacidad[[varX]], datosColombiaCapacidad[varZ])

# Obtenemos la ecuación de la recta
print("\nEcuación de la recta: y = ", round(modeloCapacidad.coef_[0],3), "x + ", round(modeloCapacidad.intercept_,3))
# Obtenemos el coeficiente de correlación
print("Coeficiente de correlación: ", round(np.corrcoef(datosColombiaCapacidad[varX], datosColombiaCapacidad[varZ])[0,1],3))
# Obtenemos el coeficiente de determinación
print("Coeficiente de determinación: ", round(r2_score(datosColombiaCapacidad[varZ], modeloGeneracion.predict(datosColombiaCapacidad[[varX]])),3))

datoPrediccion = 2024
print("\nPredicción de capacidad en el año ", datoPrediccion, ": ", modeloCapacidad.predict([[datoPrediccion]]))

# Graficar 
plt.plot(datosColombiaCapacidad[varX], datosColombiaCapacidad[varZ], label='y')
plt.plot(datosColombiaCapacidad[varX], modeloCapacidad.predict(datosColombiaCapacidad[[varX]]), label='predicción')
plt.title("Regresión lineal - Capacidad Instalada Colombia")
plt.xlabel("Capacidad Instalada (MW)")
plt.ylabel("Año")
plt.legend()
plt.grid()
plt.show()
# Regresión con sklearn ...................................................................................................

# Predicción ..............................................................................................................
# Columna de años para el nuevo dataframe, va desde 2017 hasta 2030
tiempo = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]

# Lista para almacenar los valores predecidos
prediccionGeneracion = []

# Bucle for para calcular los valores de la predicción
for valor in tiempo:
    valorPrediccion =  modeloGeneracion.predict([[valor]]) # Se predice el valor
    prediccionGeneracion.append(valorPrediccion)

# Crear el DataFrame
dfPrediccionGeneracion = pd.DataFrame({'Year': tiempo, 'Electricity Generation (GWh)': prediccionGeneracion})
print(dfPrediccionGeneracion)

# Graficar la predicción 
plt.plot(tiempo, prediccionGeneracion, label='predicción')
plt.plot(datosColombiaGeneracion[varX], datosColombiaGeneracion[varY], label='real')
plt.title("Predicción de Generación Fotovoltaica en Colombia al 2030")
plt.xlabel("Generación eléctrica (GWh)")
plt.ylabel("Año")
plt.legend()
plt.grid()
plt.show()

# Lista para almacenar los valores predecidos
prediccionCapacidad = []

# Bucle for para calcular los valores de la predicción
for valor in tiempo:
    valorPrediccion =  modeloCapacidad.predict([[valor]]) # Se predice el valor
    prediccionCapacidad.append(valorPrediccion)

# Crear el DataFrame
dfPrediccionCapacidad = pd.DataFrame({'Year': tiempo, 'Electricity Installed Capacity (MW)': prediccionCapacidad})
print(dfPrediccionCapacidad)

# Graficar la predicción 
plt.plot(tiempo, prediccionCapacidad, label='predicción')
plt.plot(datosColombiaCapacidad[varX], datosColombiaCapacidad[varZ], label='real')
plt.title("Predicción de Capacidad Instalada en Colombia al 2030")
plt.xlabel("Capacidad Instalada (MW)")
plt.ylabel("Año")
plt.legend()
plt.grid()
plt.show()
# Predicción ..............................................................................................................
