# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:19:08 2026

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


# %%===========================================================================
# # Plantilla para FIR
# # =============================================================================
#PROBAR TRANSICIONES SIMETRICAS
#probar una plantilla mas ajustada (Predistorcion) de lo que quiero para que si cumple mal tenga un changui

nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = .4 # Hz
wp1 = .5 # Hz
wp2 = 35 # Hz
ws2 = 36 # Hz

wp = np.array([wp1, wp2])
ws = np.array([ws1, ws2])

wpv = np.array([.5, 35])
wsv = np.array([.1, 45])

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
    np.logspace (start= -2, stop = 0, num = 500),     #primera transicion
    np. linspace(start = 1.26, stop = 35, num = 200),  # banda de paso
    np.logspace(start = 1.55, stop = 1.65, num = 200),#segunda transicion
    np.linspace(start = 46, stop = fs//2, num = 50)]) #banda de stop
               

numtaps = 2500
freq = np.array([0., ws1, wp1, wp2, ws2, fs//2])

# CORRECCIÓN DE GANANCIAS:
# Definimos la plantilla real en dB (0 dB en la banda de paso, -gstop en las de stop)
gains = np.array([0., 0., 1., 1., 0., 0.])


# VENTANA RECTANGULAR  par y simetrico por defecto -> FIR 2 -> CERO EN NYQ -> gains si o si con cero al final
# Forzamos que en Nyquist la ganancia lineal sea exactamente 0
gains[-1] = 0.0
ww_lineal = np.linspace(0, fs//2, 400) #PARA ASEGURAR COMPORTAMIENTO DE FIR

b_win = signal.firwin2(numtaps=numtaps, freq=freq, gain = gains, nfreqs = 2**14,  window = window, fs=fs)

w, h = signal.freqz(b_win, worN = ww , fs=fs)
z, p, k = signal.tf2zpk(b_win, a=1)


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

ax1.plot(w,20*np.log10(np.abs(h)),color='C0', label = 'Modulo')

ax1.set_ylabel("Magnitud [dB]")

ax1.grid(True)

plt.sca(ax1)
#ax1.set_xlim(0, 50)

plot_plantilla(
    filter_type = filter_type, 
    fpass = wpv,            # par [wp1, wp2] verdadero
    ripple = ripple ,
    fstop = wsv,            # par [ws1, ws2] verdadero
    attenuation = atenuacion,   
    fs = fs, 
)

ax1.legend()

# =====================================================
# FASE
# =====================================================
phase = np.unwrap(np.angle(h))

ax2.plot(w,phase,color='C1')

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
# RETARDO DE GRUPO (ax4)
# ======================================

ax4.set_title("Retardo de Grupo (FIR)") # ftype no existe para FIR, es por ventana

# Derivada discreta de la fase
gd = -np.diff(phase) / (np.diff(2 * np.pi * ww / fs)) 

# CORRECCIÓN DE TIPEO: Era np.append, no np.appened
gd = np.append(gd[0], gd) 

ax4.plot(w, gd, color='C4', linewidth=1.5)
ax4.set_xlabel("Frecuencia [Hz]")
ax4.set_ylabel("Tiempo [s]")
ax4.set_xlim(0, 50)
ax4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
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
    

