# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:06:10 2026

@author: Patog
"""

import sys
# Le agregamos la ruta exacta de la carpeta 'src' al sistema
sys.path.append(r"C:\Users\Patog\.spyder-py3\APS\ts5\pytc2\src")

# Ahora sí, hacés los imports normales
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io as sio

# E importamos desde 'pytc2' directamente
from pytc2.sistemas_lineales import plot_plantilla, group_delay

# Archivo ECG.mat.
# (variables)

# ecg_lead: Registro de ECG muestreado a 
# fs = 1kHz durante una prueba de esfuerzo
# qrs_pattern1: Complejo de ondas QRS normal
# heartbeat_pattern1: Latido normal
# heartbeat_pattern2: Latido de origen ventricular
# qrs_detections: vector con las localizaciones (en # de muestras) donde ocurren los latidos

# =========================================
# #Electrocardiograma (ECG) con ruido
# =========================================

mat_struct = sio.loadmat('ECG_TP4.mat')

# Propiedades disponibles dentro del .mat:
print(mat_struct.keys())
# Típicamente: ecg_lead, heartbeat_pattern1, heartbeat_pattern2

ecg = mat_struct['ecg_lead'].flatten()
LatidoProm = mat_struct['heartbeat_pattern1'].flatten()

fs = 1000  # Hz — frecuencia de muestreo del ECG

# Propiedades importantes:
N = len(ecg)                  # Cantidad de muestras
T = N / fs                # Duración en segundos
dt = 1 / fs              # Período de muestreo (1 ms)
tt = np.arange(N) / fs    # Eje temporal en segundos

plt.figure()
plt.plot(tt, ecg, label = 'con ruido')
plt.plot(tt, LatidoProm, label = 'promedio')

plt.label(True)
plt.title("Electrocardiograma")

# Quiero que se asemeje a los latidos promedio en cuanto a  suavidad de los trazos y nivel isoeléctrico nulo.
# mi ecg sin ruido tiene ancho de banda 32, osea que de cero a 32 contengo el 99% de la energ[ia]


# %%===========================================================================
# # Plantilla para FIR remez
# # =============================================================================
#PROBAR TRANSICIONES SIMETRICAS
#probar una plantilla mas ajustada (Predistorcion) de lo que quiero para que si cumple mal tenga un changui

nyq_frec = fs/2
ripple = .1 # dB
atenuacion = 40 # dB
 
ws1 = .05 # Hz
wp1 = .6 # Hz
wp2 = 43 # Hzr
ws2 = 43.5 # Hz

wp = np.array([wp1, wp2])
ws = np.array([ws1, ws2])

wpv = np.array([.5,35])
wsv = np.array([.1,45])


gpass = 1     # dB
gstop = 40     # dB

#window = 'Hamming'
#window = 'Hann'
#window = 'flattop'
window = 'boxcar'

#filter_type = 'lowpass'
# filter_type = 'highpass'
filter_type = 'bandpass'
# filter_type = 'bandstop'

ww = np.concat([
    np.logspace (start= -2, stop = 0.5, num = 1000),     #primera transicion
    np. linspace(start = 1.26, stop = 35, num = 100),  # banda de paso
    np.logspace(start = 1.55, stop = 1.65, num = 1000),#segunda transicion
    np.linspace(start = 46, stop = fs//2, num = 500)]) #banda de stop
               

numtaps = 2001
freq = np.array([
    0.,
    ws1, 
    wp1, 
    wp2, 
    ws2, 
    fs//2])

# CORRECCIÓN DE GANANCIAS:
# Definimos la plantilla real en dB (0 dB en la banda de paso, -gstop en las de stop)
gains = np.array([0., 1., 0.])


# VENTANA RECTANGULAR  par y simetrico por defecto -> FIR 2 -> CERO EN NYQ -> gains si o si con cero al final
# Forzamos que en Nyquist la ganancia lineal sea exactamente 0
gains[-1] = 0.0



b_win1 = signal.remez(numtaps = numtaps, bands = freq, desired = gains, weight= [1, 0.5, 0.6], type= filter_type, fs=fs)

w1, h1 = signal.freqz(b_win1, worN = ww , fs=fs)
z, p, k = signal.tf2zpk(b_win1, a=1)

# b_win = signal.firwin2(numtaps=numtaps, freq=freq, gain = gains, nfreqs = 2**14,  window = window, fs=fs)

# w, h = signal.freqz(b_win, worN = ww , fs=fs)
# z, p, k = signal.tf2zpk(b_win, a=1)


# =============================================================================
# modulo
# =============================================================================

plt.figure()
# plt.plot(w,20*np.log10(np.abs(h)),color='C0', label = 'Ventanas')
plt.plot(w1,20*np.log10(np.abs(h1)),color='C9', label = 'Remez')
plt.title(f"Respuesta en Frecuencia |numtaps = {numtaps}")
plt.ylabel("Magnitud [dB]")
plt.xlabel("Freq")
plt.grid()

plot_plantilla(
    filter_type = filter_type, 
    fpass = wpv,            # par [wp1, wp2]
    ripple = ripple ,
    fstop = wsv,            # par [ws1, ws2]
    attenuation = atenuacion,   
    fs = fs, 
)
plt.legend()
plt.show()

# %% Regiones de interés sin ruido
# -----------------------------------------------------------------------------

regs_ruido = (
    [4000, 5500], # muestras
    [10000, 11000], # muestras (reemplazado 10e3 por entero directo)
)

for ii in regs_ruido:
    # Intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
    # Creamos una figura nueva para cada región
    plt.figure(figsize=(10, 5))
    
    # Graficamos el ECG original con ruido de fondo
    plt.plot(zoom_region, ecg[zoom_region], color='navy', alpha=0.8, label='ECG Original', linewidth=2)
    
    # Evaluamos cada filtro en esta región usando fase cero
    filtros_evaluacion = ['butter', 'cheby1', 'cheby2', 'cauer']
    colores = {
    'C0',    # Azul 
    'C9',    # cyan 
    'C4',    # Morado 
    'C6'      # Rosa
}
    ecg_cr = np.reshape(ecg, (1, N))
    demora = ((numtaps-1)//2)

    # Filtrado FIR ( causal, mete 'demora' muestras de retraso )
    ecg_filtrado_fir = signal.lfilter(b_win, 1., ecg_cr)
    plt.plot(zoom_region, ecg_filtrado_fir[0, zoom_region + demora], label='FIR Boxcar (Compensado)', linewidth=1.2)
            
    plt.title(f'Zona sin Ruido: Muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Amplitud [Adimensional]')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend(loc='upper right', shadow=True)
    axes_hdl.grid(True, linestyle=':', alpha=0.5)
    # plt.yticks(()) # Descomentar si querés ocultar los valores del eje Y
    
    plt.tight_layout()
    plt.show()
 
# -----------------------------------------------------------------------------
# %% Regiones de interés con ruido (Zonas limpias)
# -----------------------------------------------------------------------------

# Corregido: se usa fs_ecg que es la variable real de tu script
regs_sin_ruido = (
    np.array([5, 5.2]) * 60 * fs,   # minutos a muestras
    np.array([12, 12.4]) * 60 * fs, # minutos a muestras
    np.array([15, 15.2]) * 60 * fs, # minutos a muestras
)

for ii in regs_sin_ruido:
    # Intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
    # Creamos una figura nueva para cada región
    plt.figure(figsize=(10, 5))
    
    # Graficamos el ECG original
    plt.plot(zoom_region, ecg[zoom_region], color='navy', label='ECG Original', linewidth=2)
    
    # Evaluamos todos los filtros para ver si deforman la señal limpia
    ecg_cr = np.reshape(ecg, (1, N))
    demora = ((numtaps-1)//2)

    # Filtrado FIR ( causal, mete 'demora' muestras de retraso )
    ecg_filtrado_fir = signal.lfilter(b_win, 1., ecg_cr)
    plt.plot(zoom_region, ecg_filtrado_fir[0, zoom_region + demora], label='FIR Boxcar (Compensado)', linewidth=1.2, color = 'C9')
            
            
    plt.title(f'Con ruido: Muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Amplitud [Adimensional]')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend(loc='upper right', shadow=True)
    axes_hdl.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    

