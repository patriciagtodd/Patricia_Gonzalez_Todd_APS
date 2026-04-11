# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:21:17 2026

@author: Patog
"""

import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    # Usamos np.arange(nn) para asegurar exactamente 'nn' muestras
    tt = np.arange(0, nn) * ts 
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx

def mi_funcion_ruido (tt, Pr):
    desvio = np.sqrt(Pr)
    r = np.random.normal(0, desvio, len(tt)) # lo del medio es el desvio estandar
    #verifico el valor de la varianza de r a ver si calcule bien
    return r

# Parámetros
A = np.sqrt(2)
dc = 0
fs = 1000
fase = 0
muestras = 1000
f0 = fs/muestras #delta f que es RESOLUCION ESPECTRAL
bits = 4
# 4 bits => 16 niveles
Vf = 2
q = Vf/(2**(bits))
Ps = (A**2)/ 2 
Pq = (q**2)/12 #para ruido de cuantización uniforme
kn = 1
Pn = kn * Pq #ESTO DEBERIA SER LA VARIANZA
SNR = Ps/Pn
SNR_db = 10*np.log10(SNR + 1e-15)

print(q/2)



# Llamadas a las funciones
tt, S = mi_funcion_sen(A, dc, f0, fase, muestras, fs)
n = mi_funcion_ruido(tt, Pn)
print(f" var = {np.var(S)}")
Sn = S + n

# recorte al rango del ADC
Sn_clip = np.clip(Sn, -Vf, Vf)

#Cuantizacion
Sq = q* np.round(Sn_clip/q)

#Error de cuantizacion
Error_q = Sq - Sn

# Gráfico senos
plt.plot(tt, S, label="S")
plt.plot(tt, Sn, label="S + n", alpha=0.7)
plt.plot(tt, Sq, label = " Sq", alpha=0.4)
plt.xlabel("t[s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()

#%% Hitoggrama del error de cuantizacion
plt.hist(Error_q, bins=10, density=True, color='purple', alpha=0.4,
         edgecolor='black')

# Media (esto sí es horizontal si querés verlo como densidad, pero suele no tener mucho sentido)
# mejor marcarla vertical también:
plt.axhline(1/q, color='C4',
            label=f"Altura teórica = {1/q:.3f}",
            linestyle='--', linewidth=2)

# Límites teóricos del error de cuantización
plt.axvline(q/2, color='m',
            label=f"+q/2 = {q/2:.3f}",
            linestyle='--', linewidth=2)

plt.axvline(-q/2, color='m',
            label=f"-q/2 = {-q/2:.3f}",
            linestyle='--', linewidth=2)

plt.grid(True)
plt.legend()
plt.show()

#%%

#XX_senal = np.fft.fft(xx)
#XX_ruido = np.fft.fft(ruido)
#estos son completos, para graficralos necesito su  modulo

#mods = np.abs(XX_senal)
#modr = np.abs(XX_ruido)

#Sr
e_Sr = np.fft.fft(Sn)/(muestras)  # escalo parra que me de donde yo quiero
mod_e_Sr = np.abs((e_Sr))**2
#Sq
e_Sq = np.fft.fft(Sq)/(muestras)
mod_e_Sq = np.abs((e_Sq))**2

# paso a db
mod_e_Sr_db = 10*np.log10(mod_e_Sr + 1e-15) #db
mod_e_Sq_db = 10*np.log10(mod_e_Sq + 1e-15) #db 

#pisos => BUSCAR EL ESTIMADO TEORICO
piso_analog = np.mean(mod_e_Sr_db) 
piso_digital = np.mean(mod_e_Sq_db)
'''
piso_analog = 10 * np.log10(np.mean(mod_e_Sr) + 1e-15)
piso_digital = 10 * np.log10(np.mean(mod_e_Sq)+ 1e-15)
'''
# Promedio de la potencia, luego logaritmo
piso_analog_TEORICO = 10 * np.log10(Pn/(muestras) + 1e-15)
piso_digital_TEORICO = 10 * np.log10(((Pq + Pn)/(muestras))+ 1e-15)

print(f" T vs M {piso_analog_TEORICO} ~ {piso_analog} analogicos")
print(f" T vs M {piso_digital_TEORICO} ~ {piso_digital} digital")

espectro = np.fft.fftfreq(muestras,1/fs)
ffreq = np.arange(0, muestras)


# Gráfico SOLO DEL DELTA
plt.figure(figsize=(10, 4))
plt.plot(ffreq, mod_e_Sr_db,'c', label= "Espectro de S + n (analog.)", linewidth=1, alpha = 0.9)
plt.plot(ffreq, mod_e_Sq_db,'b', label= "Espectro de Sq (digital)", linewidth=1, alpha = 0.9)
plt.axhline(piso_analog_TEORICO ,color = 'C6',label = f"Piso de S + n (analog.)  = {piso_analog_TEORICO}db ",
            linestyle='--', linewidth=2)
plt.axhline(piso_digital_TEORICO ,color = 'C14',label = f"Piso digital = {piso_digital_TEORICO}db",
            linestyle='--', linewidth=2)
#plt.xlim(0,500)
plt.grid(True)
plt.title(f"Comparativa de Espectros con kn = {kn}, q = {q}")
plt.legend()
plt.xlabel("freq[Hz]")
plt.ylabel("Amplitud")

#print(f" Maximo delta = {np.max(mod_e_Sr_db)})

 #%%
