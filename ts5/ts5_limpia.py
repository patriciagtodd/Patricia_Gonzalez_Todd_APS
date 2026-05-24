import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
#from scipy.io.wavfile import write


#%%###############
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

fs_ecg = 1000

# Valores de Nperseg
nperseg_vec = [250, 500, 1000, 2000]

# =========================
# TABLA
# =========================

resultados = []

for div in nperseg_vec:

    # Welch
    f, PSD = sig.welch(
        ecg_one_lead - np.mean(ecg_one_lead),
        fs=fs_ecg,
        window='hamming',
        nperseg=div,
        noverlap=div//2
    )

    # PSD en dB
    PSD_db = 10*np.log10(PSD + 1e-12)

    # -------- GRÁFICO --------
    plt.plot(f, PSD_db, label=f'Nperseg = {div}')

    # =========================
    # MÉTRICAS
    # =========================

    # Resolución espectral
    delta_f = fs_ecg / div

    # Varianza
    varianza = np.var(PSD)

    # Energía acumulada
    energia_acum = np.cumsum(PSD)

    # Energía total
    energia_total = energia_acum[-1]

    # Energía normalizada
    energia_norm = energia_acum / energia_total

    # BW 95%
    indice_bw = np.where(energia_norm >= 0.95)[0][0]

    bw_95 = f[indice_bw]

    # Guardar resultados
    resultados.append([
        div,
        delta_f,
        varianza,
        bw_95
    ])

# =========================
# CONFIGURACIÓN DEL GRÁFICO
# =========================

plt.title('Comparación PSD ECG - Método de Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)
plt.legend()

plt.show()

# =========================
# IMPRIMIR TABLA
# =========================

print(f"{'Nperseg':<10} | {'Δf [Hz]':<10} | {'Varianza':<15} | {'BW 95% [Hz]':<15}")
print("-"*65)

for r in resultados:

    print(
        f"{r[0]:<10} | "
        f"{r[1]:<10.4f} | "
        f"{r[2]:<15.6e} | "
        f"{r[3]:<15.2f}"
    )
    
#%%###############
## AUDIO CUCARACHA
##################
fs_cucaracha, cuca_wav = sio.wavfile.read('la cucaracha.wav')

plt.figure()
plt.plot(cuca_wav)
plt.title("Cucaracha")
plt.show()

#%% =========================
# TABLA
# =========================
nperseg_vec = [250, 500, 1000, 2000]
resultados = []
for div in nperseg_vec:
    f, PSD = sig.welch(
        cuca_wav - np.mean(cuca_wav),
        fs=fs_cucaracha,           # ✅ fixed: was fs_ecg
        window='hamming',
        nperseg=div,
        noverlap=div//2
    )
    PSD_db = 10*np.log10(PSD + 1e-12)
    plt.plot(f, PSD_db, label=f'Nperseg = {div}')

    delta_f = fs_cucaracha / div   # ✅ consistent with the correct fs
    varianza = np.var(PSD)
    energia_acum = np.cumsum(PSD)
    energia_total = energia_acum[-1]
    energia_norm = energia_acum / energia_total
    indice_bw = np.where(energia_norm >= 0.95)[0][0]
    bw_95 = f[indice_bw]
    resultados.append([div, delta_f, varianza, bw_95])

plt.title('Comparación PSD Audio - Método de Welch')  # ✅ fixed title
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)
plt.legend()
plt.show()

print(f"{'Nperseg':<10} | {'Δf [Hz]':<10} | {'Varianza':<15} | {'BW 95% [Hz]':<15}")
print("-"*65)
for r in resultados:
    print(
        f"{r[0]:<10} | "
        f"{r[1]:<10.4f} | "
        f"{r[2]:<15.6e} | "
        f"{r[3]:<15.2f}"
    )                              