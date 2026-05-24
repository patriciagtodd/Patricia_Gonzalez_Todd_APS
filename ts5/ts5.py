# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:01:38 2026

@author: Patog
"""

# -*- coding: utf-8 -*-
"""
Lectura  de señales
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


# %%
"""
#Electrocardiograma (ECG) con ruido

mat_struct = sio.loadmat('ECG_TP4.mat')

# Propiedades disponibles dentro del .mat:
print(mat_struct.keys())
# Típicamente: ecg_lead, heartbeat_pattern1, heartbeat_pattern2

ecg = mat_struct['ecg_lead'].flatten()
fs_ecg = 1000  # Hz — frecuencia de muestreo del ECG

# Propiedades importantes:
N_ecg = len(ecg)                  # Cantidad de muestras
T_ecg = N_ecg / fs_ecg                # Duración en segundos
dt_ecg = 1 / fs_ecg               # Período de muestreo (1 ms)
t_ecg = np.arange(N_ecg) / fs_ecg    # Eje temporal en segundos

plt.figure()
plt.plot(t_ecg, ecg)
plt.title("Electrovcardiograma")
plt.show()

print("ECG:")
print(f"Muestras: {N_ecg}")
print(f"Duración: {T_ecg:.2f} s")
print(f"fs: {fs_ecg} Hz  →  Nyquist: {fs_ecg/2} Hz")

def saber_L(x, K): 
    N = len(x)          # N lo sacás de la señal, no lo pasás como parámetro
    L = N // K
    
    if N % K != 0:
        print(f"⚠️ {N} muestras no se divide exactamente en {K} segmentos.")
        print(f"   Sobrán {N % K} muestras, se descartarán del final.")
    else:
        print(f"Longitud de segmentos: {L}")
    return L
"""
##################
## ECG sin ruido
##################


ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure()
plt.plot(ecg_one_lead)

N_ecg = len(ecg_one_lead)

ecg = np.reshape(ecg_one_lead , (1, N_ecg))



"""# %% 
# Pletismografía (PPG.csv)

ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)
if ppg.ndim > 1:
    ppg = ppg[:, 0]           # Tomar solo la primera columna

fs_ppg = 400  # Hz

N_ppg = len(ppg)
T_ppg = N_ppg / fs_ppg
t_ppg = np.arange(N_ppg) / fs_ppg

print("PPG:")
print(f"Muestras: {N_ppg}")
print(f"Duración: {T_ppg:.2f} s")

plt.figure()
plt.plot(t_ppg, ppg)
plt.title("Pletismografía")
plt.show()
"""

"""# %% Audio 
# Lectura de audios 

# Cargar el archivo CSV como un array de NumPy
fs_cucaracha, cuca_wav = sio.wavfile.read('la cucaracha.wav')
fs_prueba_au, prueba_wav = sio.wavfile.read('prueba psd.wav')
fs_silvido, silva_wav = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(cuca_wav)
plt.title("Cuacaracha")
plt.show()

plt.figure()
plt.plot(prueba_wav)
plt.title("Audio prueba")
plt.show()

plt.figure()
plt.plot(silva_wav)
plt.title("Silvido")
plt.show()

# si quieren oirlo, tienen que tener el siguiente módulo instalado
#pip install sounddevice
#import sounddevice as sd
#sd.play(wav_data, fs_audio)
"""

# %% Funciones de estimación espectral HECHOS POR MI

# Periodograma
"""
def Periodograma (x, fs, N): #FFT  CLASICA(?)
    
    X = np.fft.fft(x)/(N)  
    pot = np.abs((X))**2
    #pot[1:-1] *= 2   # duplicar excepto DC y Nyquist
    ffreq = np.arange(0, N)*(fs/N)
    
    return pot,ffreq

def Bartlett(x,fs,N, K):
    L = N // K if N % K == 0 else None
    suma_potencias = 0
    if not L:
        print(f"⚠️ ¡Error! {N} no se puede dividir exactamente en {K} segmentos.")
        return False, None  # Retorna False para avisar que falló
    else:
        for i in range(K):
            pot, ffreq = Periodograma(x[i*L: (i+1)*L],fs, L)
            suma_potencias += pot
        bartlett = suma_potencias/K
        ffreq_b = np.arange(0, N)*(fs/N)
    return bartlett, ffreq_b

# %% correciones claudia


def Periodograma(x, fs):          # N sobra como parámetro, lo sacás de x
    N = len(x)                    # siempre consistente con el segmento actual
    
    X = np.fft.rfft(x) / N       # rfft: solo freqs positivas (señal real)
    pot = np.abs(X)**2
    pot[1:-1] *= 2                # ← descomentá esto, es necesario para energía correcta
    
    ffreq = np.fft.rfftfreq(N, d=1/fs)   # más limpio que arange manual
    
    return pot, ffreq


def Bartlett(x, fs, K):          # N sobra acá también
    N = len(x)
    L = N // K
    
    if N % K != 0:
        print(f"⚠️ {N} muestras no es divisible exactamente por {K}.")
        print(f"   Se descartan las últimas {N % K} muestras.")
        x = x[:K*L]              # recortar en lugar de abortar, más práctico
    
    suma_potencias = 0
    
    for i in range(K):
        segmento = x[i*L : (i+1)*L]
        pot, ffreq = Periodograma(segmento, fs)   # L se calcula adentro con len(x)
        suma_potencias += pot
    
    bartlett = suma_potencias / K
    # ffreq viene del último segmento, todos tienen el mismo eje
    
    return bartlett, ffreq


#freqs, psd = sig.welch(x, fs=fs, window='hamming', nperseg=L, noverlap=L//2)
"""


"""
De hecho Bartlett es un caso especial de Welch:
python# Bartlett = Welch con ventana rectangular y sin overlap
freqs, psd_bartlett = sig.welch(x, fs=fs, window='boxcar',
                                 nperseg=L, noverlap=0)

# Welch clásico = ventana Hamming/Hann con 50% overlap  
freqs, psd_welch = sig.welch(x, fs=fs, window='hamming',
                              nperseg=L, noverlap=L//2)
"""
"""
def blackman_tukey(x,  M = None):    
    
    # N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px;
"""

# %% WELCH

# Valores de división que quieres probar
L = [1000, 2000, 3000, 5000]

for div in L:

    # Calcular L según el valor elegido
    #L = saber_L(ecg, div)

    # Welch
    ffreq, E_ECG = sig.welch(ecg_one_lead ,fs_ecg,window='hamming',nperseg=div,noverlap=div//2)

    # Pasar a dB
    E_ECG_db = 10*np.log10(E_ECG + 1e-12)

    # Varianza
    var_ecg = np.var(E_ECG)

    print(f"Nperseg = {L}  |  Varianza = {var_ecg}")

    # Agregar curva al mismo gráfico
    plt.semilogy(ffreq, E_ECG_db, label=f'Nperseg={L}')

# Configuración final del gráfico
plt.title('Comparación PSD ECG - Método de Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [V²/Hz]')
plt.grid()
plt.legend()

plt.show()


