import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo # Para exportar el gráfico en html
import json
import altair as alt

with open('Mapas/NASA.geojson', 'r') as f:
  paises = json.load(f)


df = pd.read_csv('Datos/TestDataset.csv', sep=';')

locs = df['Country']

for loc in paises['features']:
    loc['id'] = loc['properties']['name']
mapa = go.Figure(go.Choroplethmapbox(
                 geojson=paises,
                 locations=locs,
                 z=df['Number'],
                 colorscale='Viridis',
                 colorbar_title="Número"))
mapa.update_layout(mapbox_style="cartro-positron")

#mapa.show()
pyo.plot(mapa, filename='TestMapa.html')