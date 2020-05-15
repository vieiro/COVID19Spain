#!/usr/bin/env python3

import requests
import pandas
import os
import time
from datetime import datetime, timedelta, date
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
import scipy.stats as stats
import numpy

## Covid19 en España
## Modelo SIR
## Ver https://en.wikipedia.org/wiki/Mathematical_modelling_of_infectious_disease#The_SIR_model

## Constantes

# URL para descargar datos
NACIONAL_URL = "https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/nacional_covid19.csv"

# Nombre de fichero caché
NACIONAL_FILE = "nacional_covid19.csv"

# Columnas del archivo CSV
COL_FECHA = 'fecha'
COL_CASOS_TOTAL = 'casos_total'
COL_CASOS_PCR = 'casos_pcr'
COL_ALTAS = 'altas'
COL_MUERTES = 'fallecimientos'
COL_UCI = 'ingresos_uci'
COL_HOSPITAL = 'hospitalizados'

## Funciones

def download(url, filename):
    """
    Descarga 'url' a 'filename' sólo si hace más de 10 minutos que se ha descargado.
    """
    last_modified = 0
    if os.path.isfile(filename):
        last_modified = os.path.getmtime(filename)
    now = time.time()
    difference = now - last_modified
    if difference >= 10*60:
        print('Last modified', difference, 'proceeding with download', url)
        request = requests.get(url, allow_redirects = True)
        open(filename, 'wb').write(request.content)

def readCSV():
    """
    Lee el archivo CSV
    """
    DATE_COLUMNS = [COL_FECHA]
    COLUMN_TYPES = {
            COL_FECHA: 'str', 
            COL_CASOS_TOTAL: 'float64',
            COL_ALTAS: 'float64',
            COL_MUERTES: 'float64',
            COL_UCI: 'float64',
            COL_HOSPITAL: 'float64'
            }
    data = pandas.read_csv(NACIONAL_FILE, sep=',', dtype = COLUMN_TYPES, parse_dates=DATE_COLUMNS)
    # Convertir celdas vacías a 0
    data = data.fillna(0)
    return data


## Descargar fichero
download(NACIONAL_URL, NACIONAL_FILE)

## Leer el fichero
data = readCSV()

## Convertir los timestamps a fechas para pintar
x = data[COL_FECHA].apply(lambda x: x.date())

## Obtener los casos totales, las altas y los fallecimientos
casos_total = data[COL_CASOS_TOTAL]
casos_pcr = data[COL_CASOS_PCR]
altas = data[COL_ALTAS]
muertes = data[COL_MUERTES]

## Preparar los datos del modelo SIR:

## I: Infectados acumulados, esto es igual al número de casos reportados
I = casos_total + casos_pcr

## R: Recuperados, esto es la suma de las altas y los fallecimientos
R = altas + muertes 

## Los Infectados actuales son los infectados acumulados menos los que se han recuperado y los que han muerto en el día anterior...

I = I - R

## Calcular R0

DI = numpy.gradient(I)
DR = numpy.gradient(R)

R0 = (DI + DR)/DR
# Las divisiones por cero indican un R0 grande
R0[numpy.isnan(R0)] = 1000

#DI = numpy.diff(I)
#DR = numpy.diff(R)
#x = x[1:]
#
#R0 = []
#for i in range(0, len(DI)):
#    if DR[i] == 0:
#        R0.append(1000)
#    else:
#        R0.append((DI[i])/DR[i])

## Preparar el gráfico
pyplot.style.use('Solarize_Light2')
# Usar IBM Plex Sans, que me gusta

fontfamily = 'IBM Plex Sans'
font = {'family': fontfamily,
        'weight': 'normal',
        'size': 10}
pyplot.rc('font', **font)
palette = pyplot.get_cmap('Dark2')


## En el eje izquierdo 'subplot' pintaremos I y R.
fig, subplot = pyplot.subplots(1,1, figsize=(9.3, 6.8)) 

subplot.set_xlabel('Fecha')
subplot.set_ylabel('R0 (t)')
subplot.set_ylim([0,5])
subplot.set_xlim(date(2020, 3, 9), datetime.now())
subplot.tick_params(axis='x', color='black', labelsize='11', labelcolor='black')
subplot.tick_params(axis='y', color='black', labelsize='11', labelcolor='black')
subplot.xaxis_date()
subplot.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
subplot.xaxis.set_major_locator(mdates.WeekdayLocator())
subplot.xaxis.set_minor_locator(mdates.DayLocator())
lastR0 = R0[-1]
lastR0 = f'Último R0 {lastR0:.2f}'
pyplot.suptitle("COVID-19 España. Modelo SIR. R0 (t)\n" + lastR0, fontweight='bold', color='black', fontsize=16) #fontsize=12, fontweight=400, color='black')
hoy = datetime.now().strftime('%d-%m-%Y')
pyplot.figtext(0.01, 0.02, 'Fuente: https://github.com/datadista/datasets/blob/master/COVID%2019/nacional_covid19.csv ' + hoy,fontsize=12, color='gray')
subplot.plot(x, R0, linewidth=1.9, color='red', alpha=1, label='R0')
pyplot.savefig('covid-spain-sir-r0.png')
pyplot.show()

print('R0', R0)




