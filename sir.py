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
COL_CASOS = 'casos'
COL_ALTAS = 'altas'
COL_MUERTES = 'fallecimientos'
COL_UCI = 'ingresos_uci'
COL_HOSPITAL = 'hospitalizados'

# Constante N: Número de personas infectables
# En España somos unos 46 millones de personas
N = 46000000

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
            COL_CASOS: 'float64',
            COL_ALTAS: 'float64',
            COL_MUERTES: 'float64',
            COL_UCI: 'float64',
            COL_HOSPITAL: 'float64'
            }
    data = pandas.read_csv(NACIONAL_FILE, sep=',', dtype = COLUMN_TYPES, parse_dates=DATE_COLUMNS)
    # Convertir celdas vacías a 0
    data = data.fillna(0)
    return data

## Suma
def sumSeries(a,b):
    """
    Dadas dos series pandas, las suma
    """
    result = []
    for index, aa in a.iteritems():
        result.append(aa + b.iloc[index])
    return result

## Calcula la "derivada" de un vector.
def dFdt(F):
    """
    Calcula la derivada de un vector.
    Se aproxima la derivada por incrementos de la función en un
    período de tiempo "dt" que es, obviamente, un día.
    O sea: dFdt ~ (F(t+1) - F(t)) / ((t+1)-t) = F(t+1) - F(t)
    """
    dfdt = [0]
    for i in range(1,len(F)):
        if i == 0:
            dfdt.append(F[1]-F[0])
        elif i == len(F)-1:
            dfdt.append(F[len(F)-1]-F[len(F)-2])
        else:
            dfdt.append(0.5*(F[i+1] - F[i-1]))
    return dfdt

## Conversión de fechas
def dates(x):
    """
    Dada una serie pandas de timestamps, esta
    función convierte los timestamps a fechas.
    """
    xx = []
    for index, date in x.iteritems():
        xx.append(date.date())
    return xx

## Cálculo de parámetros del modelo SIR
# (1) N = S + I + R  (N: Número total de personas)
# (2) dS/dt = - beta S I / N
# (3) dI/dt = beta S I / N - gamma I
# (4) dR/dt = gamma I
# (5) beta : Coeficiente de infección (Un enfermo contagia a beta * N otras personas por unidad de tiempo)
# (6) gamma :  Coeficiente de recuperación (recuperado o fallecido)
def modeloSIR(x, I, R, S, dIdt, dRdt, dSdt):
    # (4): dR/dt = gamma I => gamma = dR/dt / I
    gammas = []
    xx = []
    gamma_by_date = {}
    for i in range(0, len(I)):
        ii = I[i]
        d = dRdt[i]
        if ii != 0 and d != 0:
            g = dRdt[i] / ii
            gammas.append(g)
            xx.append(x[i])
            gamma_by_date[x[i]] = g

    # Cálculo del valor medio y error estándar
    stderr = numpy.std(gammas)
    media = numpy.mean(gammas)
    print('Gamma promedio: ', media, ' error estándar ', stderr)
    gamma_min = []
    gamma_avg = []
    gamma_max = []
    GAMMA_PROMEDIO = media
    for i in range(len(xx)):
        gamma_avg.append(media)
        gamma_min.append(media - stderr)
        gamma_max.append(media + stderr)

    # (3) dI/dt = beta S I / N - gamma I
    # (4) dR/dt = gamma I
    # Entonces
    # dI/dt = beta S I / N - dRdt
    # dI/dt + dRdt = beta S I / N
    # [ dI/dt + dRdt ] / I = beta S / N
    # beta = [ dI/dt + dRdt ] / I / S * N
    betas = []
    xxx = []
    beta_by_date = {}
    for i in range(len(I)):
        ss = S[i]
        ii = I[i]
        if ss != 0 and ii != 0:
            beta = N * (dIdt[i] + dRdt[i]) / ss / ii
            betas.append(beta)
            beta_by_date[x[i]] = beta
            xxx.append(x[i])
    stderr = numpy.std(betas)
    media = numpy.mean(betas)
    print('Betas', betas)
    print('Beta promedio: ', media, ' error ' , stderr)
    beta_min = []
    beta_avg = []
    beta_max = []
    for i in range(len(xxx)):
        beta_avg.append(media)
        beta_min.append(media - stderr)
        beta_max.append(media + stderr)
    BETA_PROMEDIO = media

    # R0 = BETA / GAMMA
    xxxx = []
    R0 = []
    for i in range(0, min(len(betas), len(gammas))):
        thex = xx[i]
        beta = beta_by_date[thex]
        gamma = gamma_by_date[thex]
        if gamma != 0:
            xxxx.append(thex)
            R0.append(beta / gamma)

    return xx, gammas, gamma_min, gamma_avg, gamma_max, xxx, betas, beta_min, beta_avg, beta_max, xxxx, R0



## Descargar fichero
download(NACIONAL_URL, NACIONAL_FILE)

## Leer el fichero
data = readCSV()

## Convertir los timestamps a fechas para pintar
x = dates(data[COL_FECHA])

## Obtener los casos totales, las altas y los fallecimientos
casos = data[COL_CASOS]
altas = data[COL_ALTAS]
muertes = data[COL_MUERTES]

## Preparar los datos del modelo SIR:

## I: Infectados, esto es igual al número de casos reportados
I = casos

## R: Recuperados, esto es la suma de las altas y los fallecimientos
R = sumSeries(altas, muertes)

## S: Los que no son ni I ni R (los "infectables")
S = []
for index in range(len(I)):
    S.append(N - I[index] - R[index])

## dI/dt: Aproximación de la derivada de I con incrementos diarios en I
dIdt = dFdt(I)

## dR/dt: Aproximación de la derivada de R con incrementos diarios en R
dRdt = dFdt(R)

## dS/dt: Aproximación de la derivada de S con incrementos diarios en S
dSdt = dFdt(S)

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

subplot.xaxis_date()
subplot.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
subplot.xaxis.set_major_locator(mdates.WeekdayLocator())
subplot.xaxis.set_minor_locator(mdates.DayLocator())
subplot.tick_params(axis='x', color='black', labelsize='11', labelcolor='black')
subplot.tick_params(axis='y', color='black', labelsize='11', labelcolor='black')
subplot.set_xlabel('Fecha')
subplot.set_ylabel("I, R")
subplot.plot(x, I, linewidth=1.9, color='red', alpha=1, label='I')#, linewidth=1.9, label='Nuevos casos diarios') 
subplot.plot(x, R, linewidth=1.9, color='green', alpha=1, label='R')#, linewidth=1.9, label='Nuevos casos diarios') 

## En el eje derecho 'derivadas' pintaremos dI/dt y dR/dt.
derivadas = subplot.twinx()
derivadas.tick_params(axis='y', color='blue', labelsize='11', labelcolor='black')
derivadas.set_ylabel("dI/dt y dR/dt")
derivadas.plot(x, dIdt, linewidth=1.9, color='red', linestyle=':', alpha=1, label='dI/dt')#, linewidth=1.9, label='Nuevos casos diarios') 
derivadas.plot(x, dRdt, linewidth=1.9, color='green', linestyle=':', alpha=1, label='dR/dt')#, linewidth=1.9, label='Nuevos casos diarios') 
derivadas.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
derivadas.xaxis.set_major_locator(mdates.WeekdayLocator())
derivadas.xaxis.set_minor_locator(mdates.DayLocator())

## Cosas generales
pyplot.suptitle("COVID-19 España. Modelo SIR.", fontweight='bold', color='black', fontsize=16) #fontsize=12, fontweight=400, color='black')
subplot.legend()
derivadas.legend()

pyplot.savefig('covid-spain-sir.png')
# pyplot.show()

## Calcular parámetros del modelo SIR
x, gamma, gmin, gavg, gmax, xx, beta, beta_min, beta_avg, beta_max, xxxx, R0= modeloSIR(x, I, R, S, dIdt, dRdt, dSdt)

fig, subplot = pyplot.subplots(1,1, figsize=(9.3, 6.8)) 
subplot.set_xlabel('Fecha')
subplot.xaxis_date()
betaplot = subplot.twinx()

subplot.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
subplot.xaxis.set_major_locator(mdates.WeekdayLocator())
subplot.xaxis.set_minor_locator(mdates.DayLocator())
subplot.tick_params(axis='x', color='black', labelsize='11', labelcolor='black')
subplot.tick_params(axis='y', color='red', labelsize='11', labelcolor='red')
subplot.set_ylabel('Gamma', color='red', fontsize=14)
subplot.plot(x, gamma, linewidth=1.9, color='red', linestyle=':', alpha=0.8, label='Gamma')#, linewidth=1.9, label='Nuevos casos diarios') 
subplot.plot(x, gavg, linewidth=1.9, color='red', alpha=1, label='Gamma Promedio')#, linewidth=1.9, label='Nuevos casos diarios') 
subplot.fill_between(x, gmin, gmax, alpha=0.1, facecolor='red')

betaplot.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
betaplot.xaxis.set_major_locator(mdates.WeekdayLocator())
betaplot.xaxis.set_minor_locator(mdates.DayLocator())
betaplot.tick_params(axis='y', color='blue', labelsize='11', labelcolor='blue')
betaplot.set_ylabel('Beta', color='blue', fontsize=14)
betaplot.plot(xx, beta, linewidth=1.9, color='blue', linestyle=':', alpha=0.8, label='Beta')#, linewidth=1.9, label='Nuevos casos diarios') 
betaplot.plot(xx, beta_avg, linewidth=1.9, color='blue', alpha=1, label='Beta Promedio')#, linewidth=1.9, label='Nuevos casos diarios') 
betaplot.fill_between(xx, beta_min, beta_max, alpha=0.1, facecolor='blue')

pyplot.suptitle("COVID-19 España. Modelo SIR. Gamma, Beta.", fontweight='bold', color='black', fontsize=16) #fontsize=12, fontweight=400, color='black')
pyplot.savefig('covid-spain-sir-beta-gamma.png')
pyplot.show()

# Gráfico de R0

fig, subplot = pyplot.subplots(1,1, figsize=(9.3, 6.8)) 
subplot.set_xlabel('Fecha')
subplot.xaxis_date()
subplot.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
subplot.xaxis.set_major_locator(mdates.WeekdayLocator())
subplot.xaxis.set_minor_locator(mdates.DayLocator())
subplot.tick_params(axis='x', color='black', labelsize='11', labelcolor='black')
subplot.tick_params(axis='y', color='red', labelsize='11', labelcolor='red')
subplot.plot(xxxx, R0, linewidth=1.9, color='red', alpha=1, label='R0')
subplot.set_ylim([0,15])
subplot.set_ylabel('R0')

pyplot.suptitle("COVID-19 España. Modelo SIR. R0.", fontweight='bold', color='black', fontsize=16) #fontsize=12, fontweight=400, color='black')
pyplot.savefig('covid-spain-sir-r0.png')
pyplot.show()

print('R0', R0)




