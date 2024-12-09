import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo # Para exportar el gráfico en html
import json
from urllib.request import urlopen

with urlopen('Mapas/custom.json') as response:
    paises = json.load(response)

df = pd.read_csv('TestDataset.csv', sep=';')
print(df.head())