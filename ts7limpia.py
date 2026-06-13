# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:33:26 2026

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
fs = 1000  # Hz — frecuencia de muestreo del ECG

# Propiedades importantes:
N = len(ecg)                  # Cantidad de muestras
T = N / fs                # Duración en segundos
dt = 1 / fs              # Período de muestreo (1 ms)
tt = np.arange(N) / fs    # Eje temporal en segundos

plt.figure()
plt.plot(tt, ecg)

plt.title("Electrocardiograma")

# Quiero que se asemeje a los latidos promedio en cuanto a  suavidad de los trazos y nivel isoeléctrico nulo.
# mi ecg sin ruido tiene ancho de banda 32, osea que de cero a 32 contengo el 99% de la energ[ia]


#%% =============================================================================
# # Plantilla PARA IIR
# =============================================================================

 
nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = .1 # Hz
wp1 = 1 # Hz
wp2 = 30 # Hz
ws2 = 40 # Hz

wp = np.array([wp1, wp2])
ws = np.array([ws1, ws2])

gpass = 1     # dB
gstop = 40     # dB

#ftype='cheby1'
#ftype= 'cheby2'
ftype= 'butter'
#ftype= 'cauer'

# Requerimientos de plantilla

#filter_type = 'lowpass'
# filter_type = 'highpass'
filter_type = 'bandpass'
# filter_type = 'bandstop'
 
# =============================================================================
# Grilla de evaluacion a medida -> donde poner densidad y donde no
# =============================================================================
ww = np.concat([
    np.logspace (start= -2, stop = 0, num = 200),     #primera transicion
    np. linspace(start = 1.26, stop = 35, num = 50),  # banda de paso
    np.logspace(start = 1.55, stop = 1.65, num = 100),#segunda transicion
    np.linspace(start = 46, stop = fs//2, num = 50)]) #banda de stop
               

sos = signal.iirdesign( #ANOTAR QUE HACE
    wp,
    ws,
    gpass,
    gstop,
    analog=False,
    ftype=ftype,
    output='sos',
    fs=fs)

# ======================================
# RESPUESTA EN FRECUENCIA
# ======================================

freqs, resp_freq = signal.sosfreqz(
    sos,
    worN=ww,
    fs=fs
)

# ======================================
# POLOS Y CEROS
# ======================================

z, p, k = signal.sos2zpk(sos)

# ======================================
# FIGURA
# ======================================

fig = plt.figure(figsize=(11,8))

gs = fig.add_gridspec(2,2)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
ax3 = fig.add_subplot(gs[0,1])
ax4 = fig.add_subplot(gs[1,1])

# =============================================================================
# modulo
# =============================================================================

ax1.set_title("Respuesta en Frecuencia")

ax1.plot(
    ww,
    20*np.log10(np.abs(resp_freq)),
    color='C0'
)

ax1.set_ylabel("Magnitud [dB]")

ax1.grid(True)

plt.sca(ax1)

plot_plantilla(
    filter_type = filter_type, 
    fpass = wp,            # par [wp1, wp2]
    ripple = ripple ,
    fstop = ws,            # par [ws1, ws2]
    attenuation = atenuacion,   
    fs = fs, 
)
  
ax1.legend()
# =====================================================
# FASE
# =====================================================

phase = np.unwrap(np.angle(resp_freq))

ax2.plot(
    freqs,
    phase,
    color='C1'
)

ax2.set_ylabel("Fase [rad]")
ax2.set_xlabel("Frecuencia [Hz]")

ax2.grid(True)


# =====================================================
# PLANO Z
# =====================================================

ax3.set_title("Plano Z")

# Circunferencia unitaria
theta = np.linspace(0, 2*np.pi, 1000)

ax3.plot(
    np.cos(theta),
    np.sin(theta),
    'k--',
    alpha=0.7
)

# Ceros
ax3.scatter( np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='C6', s=100, label='Ceros')

# Polos
ax3.scatter( np.real(p), np.imag(p), marker='x', color='C4', s=100, label='Polos')

ax3.axhline(0, color='black', linewidth=0.8)
ax3.axvline(0, color='black', linewidth=0.8)

ax3.set_xlabel("Parte Real")
ax3.set_ylabel("Parte Imaginaria")

ax3.set_aspect('equal', adjustable='box')

ax3.grid(True)

ax3.legend()

# ======================================
# RETARDO DE GRUPO
# ======================================

# SOS -> Transfer Function
b, a = signal.sos2tf(sos)

#w_gd, gd = signal.group_delay((b, a), w=2048,fs=fs) intentar no usarla porque hace el calculo pero no se bien donde evalua

gd = -np.diff(phase) / (np.diff(ww) * 2 * np.pi) #rad/hz = s*2pi as[i que escalo para que quede  en s


# Grilla de frecuencias ajustada al punto medio de cada diferencial
ww_gd = ww[:-1] + np.diff(ww) / 2


ax4.set_title("Retardo de Grupo")

ax4.plot(ww_gd, gd, color='C4')

ax4.set_xlabel("Frecuencia [Hz]")
ax4.set_ylabel("Tiempo [s]")

ax4.grid(True)

plt.tight_layout()

plt.show()

# -----------------------------------------------------------------------------
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
    'butter': 'C0',    # Azul 
    'cheby1': 'C9',    # cyan 
    'cheby2': 'C4',    # Morado 
    'cauer': 'C6'      # Rosa
}
    for f in filtros_evaluacion:
        try:
            sos_loop = signal.iirdesign(wp, ws, gpass, gstop, analog=False, ftype=f, output='sos', fs=fs)
            yy_fase_cero = signal.sosfiltfilt(sos_loop, ecg)
            
            plt.plot(zoom_region, yy_fase_cero[zoom_region], label=f'{f.upper()} (Fase Cero)', linewidth=1.5, color = colores.get(f))
        except Exception:
            pass # Si alguno falla por inestabilidad, continúa con los otros
            
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
    for f in filtros_evaluacion:
        try:
            sos_loop = signal.iirdesign(wp, ws, gpass, gstop, analog=False, ftype=f, output='sos', fs=fs)
            yy_fase_cero = signal.sosfiltfilt(sos_loop, ecg)
            
            plt.plot(zoom_region, yy_fase_cero[zoom_region], label=f'{f.upper()} (Fase Cero)', linewidth=1.2, color = colores.get(f))
        except Exception:
            pass
            
    plt.title(f'Con ruido: Muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Amplitud [Adimensional]')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend(loc='upper right', shadow=True)
    axes_hdl.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    

