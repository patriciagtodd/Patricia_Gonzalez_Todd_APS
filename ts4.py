
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows


def mi_funcion_ruido (N, Pr):
    desvio = np.sqrt(Pr)
    r = np.random.normal(0, desvio, N) # lo del medio es el desvio estandar
    #verifico el valor de la varianza de r a ver si calcule bien
    return r

    
# Parámetros
a0 = np.sqrt(2)
N = 1000
fs = 1000
deltaf = fs/N
W0 = np.pi/2
SNR_db = 10
Ps = (a0**2)/ 2
Pr = Ps / (10**(SNR_db/10))
R = 200

#genero las senoidales:

nn = np.arange (N)*(1/fs)
nn = nn.reshape(N,1)
matriz_n = np.tile(nn,(1,R))



fr = np.random.uniform(-2, 2, R)


f1 = (N/4 + fr) * deltaf


matriz_f = np.tile(f1, (N,1))



matriz_S = a0 * np.sin(2*np.pi* matriz_f *matriz_n) # esto es una matriz de N muestras por R de


r = mi_funcion_ruido((N,R), Pr)


matriz_Sr = matriz_S + r

# %% SIN VENTANA

w1 = np.ones(N)
w1 = w1.reshape(N,1)
w1 = np.tile(w1, (1,R))

x1 = np.fft.fft(matriz_Sr,n=N,axis=0)/(N)  # normalizada con las muestras

mag_x1 = np.abs(x1[:N//2, :]) * 2 # Espectro unilateral corregido

#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
pot_x1 = (np.abs(x1)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
pot_x1_db = 10*np.log10(pot_x1[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf


'''HASTA ACA SE QUE ESTOY ENTRE -2 Y 2'''


# %% VENTANA FLATTOP

w2 = windows.flattop(N)
w2 = w2.reshape(N,1)
w2 = np.tile(w2, (1,R))

x2 = np.fft.fft((matriz_Sr*w2),n=N,axis=0)/(N)  # normalizada con las muestras

mag_x2 = np.abs(x2[:N//2, :]) * 2 # Espectro unilateral corregido

#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
pot_x2 = (np.abs(x2)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
pot_x2_db = 10*np.log10(pot_x2[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf


'''HASTA ACA SE QUE ESTOY ENTRE -2 Y 2'''

# %% VENTANA Blackmanharris

w3 = windows.blackmanharris(N)
w3 = w3.reshape(N,1)
w3 = np.tile(w3, (1,R))

x3 = np.fft.fft((matriz_Sr*w3),n=N,axis=0)/(N)  # normalizada con las muestras

mag_x3 = np.abs(x3[:N//2, :]) * 2 # Espectro unilateral corregido

#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
pot_x3 = (np.abs(x3)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
pot_x3_db = 10*np.log10(pot_x3[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf

# %% VENTANA Hann

w4 = windows.hann(N)
w4 = w4.reshape(N,1)
w4 = np.tile(w4, (1,R))

x4 = np.fft.fft((matriz_Sr*w4),n=N,axis=0)/(N)  # normalizada con las muestras

mag_x4 = np.abs(x4[:N//2, :]) * 2 # Espectro unilateral corregido

#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
pot_x4 = (np.abs(x4)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
pot_x4_db = 10*np.log10(pot_x4[:N//2, :] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf

# %% Grafico comparativo con zoom en los maximos

fig, axes = plt.subplots(2, 4, figsize=(18, 8)) 

# Desempaquetamos usando .flatten() para asignar cada celda a tus variables de forma lineal
Rec, Flat, Black, Hann, Recz, Flatz, Blackz, Hannz = axes.flatten()

# Creamos el vector de frecuencias correcto para el espectro unilateral (500 puntos)
ffreq_unilateral = np.arange(0, N//2) * deltaf

# ==========================================
# FILA 1: ESPECTRO COMPLETO (0 a fs/2)
# ==========================================

# --- Ventana Rectangular ---
Rec.plot(ffreq_unilateral, pot_x1_db) 
Rec.set_xlim(0, fs/2)
Rec.set_title("Espectro Completo\nventana Rectangular")
Rec.grid(True)

# --- Ventana Flattop ---
Flat.plot(ffreq_unilateral, pot_x2_db) 
Flat.set_xlim(0, fs/2)
Flat.set_title("Espectro Completo\nventana Flattop")
Flat.grid(True)

# --- Ventana Blackmanharris ---
Black.plot(ffreq_unilateral, pot_x3_db) 
Black.set_xlim(0, fs/2)
Black.set_title("Espectro Completo\nventana Blackmanharris")
Black.grid(True)

# --- Ventana Hann ---
Hann.plot(ffreq_unilateral, pot_x4_db) 
Hann.set_xlim(0, fs/2) # CORREGIDO: Era set_xlim, no set_lim
Hann.set_title("Espectro Completo\nventana Hann")
Hann.grid(True)


# ==========================================
# FILA 2: ZOOM EN LOS MÁXIMOS
# ==========================================

# --- Ventana Rectangular (Zoom) ---
Recz.plot(ffreq_unilateral, pot_x1_db) 
Recz.set_xlim(N/4 * deltaf - 4*deltaf, N/4 * deltaf + 4*deltaf) 
Recz.set_ylim(-40, 20) 
Recz.set_title("Zoom Máximos\nventana Rectangular")
Recz.grid(True)

# --- Ventana Flattop (Zoom) ---
Flatz.plot(ffreq_unilateral, pot_x2_db) 
Flatz.set_xlim(N/4 * deltaf - 6*deltaf, N/4 * deltaf + 6*deltaf)
Flatz.set_ylim(-40, 20)
Flatz.set_title("Zoom Máximos\nventana Flattop")
Flatz.grid(True)

# --- Ventana Blackmanharris (Zoom) ---
Blackz.plot(ffreq_unilateral, pot_x3_db) 
Blackz.set_xlim(N/4 * deltaf - 6*deltaf, N/4 * deltaf + 6*deltaf)
Blackz.set_ylim(-40, 20)
Blackz.set_title("Zoom Máximos\nventana Blackmanharris")
Blackz.grid(True)

# --- Ventana Hann (Zoom) ---
Hannz.plot(ffreq_unilateral, pot_x4_db) 
Hannz.set_xlim(N/4 * deltaf - 6*deltaf, N/4 * deltaf + 6*deltaf)
Hannz.set_ylim(-40, 20)
Hannz.set_title("Zoom Máximos\nventana Hann")
Hannz.grid(True)

plt.tight_layout()
plt.show()
# %% Espectro de las ventanas 
W1 = np.fft.fft(np.ones(N), n=N*8)
W2 = np.fft.fft(windows.flattop(N), n=N*8)
W3 = np.fft.fft(windows.blackmanharris(N), n=N*8)

W1_db = 20*np.log10(np.abs(W1)/N + 1e-12)
W2_db = 20*np.log10(np.abs(W2)/N + 1e-12)
W3_db = 20*np.log10(np.abs(W3)/N + 1e-12)

plt.plot(W1_db, label='Rectangular')
plt.plot(W2_db, label='Flattop')
plt.plot(W3_db, label='Blackman-Harris')
plt.xlim(0, 200)
plt.legend()
plt.grid()
plt.show()

# %% Estimadores

#para mi primer estimador lo que quiero es ver que hay en la feta de omega 0
a1_x1 = mag_x1[int(N/4), :]
a1_x2 = mag_x2[int(N/4), :]
a1_x3 = mag_x3[int(N/4), :]

#quiero llegar a una matriz de rx4 de ahi hago el histograma, ese 4 son las 4 ventanas que quiero probar

'''
SEGUNDO ESTIMADOR -> AHORA TRABAJO EL ARGUMENTO
quiero quedarme con los maximos, osea todos los argumosntos N/4
aca llo que quiero es ver todos los maximos
hay lit una funcion que es argmax  y ese va a ser mi estimador (omega sombrerito),
 luego el sesgo es omega sombrerito menos omega
 quiero detectar esos maximos y por si acaso verificar que  esten bien
'''
sigma1_x1 = np.argmax(mag_x1, axis = 0)*deltaf
sigma1_x2 = np.argmax(mag_x2 , axis = 0)*deltaf
sigma1_x3 = np.argmax(mag_x3 , axis = 0)*deltaf

Mu_a = np.mean(x1)

sesgo = Mu_a - a0

#%% Hitogramas

fig, (Ha, Hf) = plt.subplots(1, 2, figsize=(14, 5))

# amplitud 
Ha.hist(a1_x1, bins=20, density=True, alpha=0.7, edgecolor='black', label = "Estimador 1 ventana unitaria")
Ha.hist(a1_x2, bins=20, density=True, alpha=0.7, edgecolor='black', label = "Estimador 1 ventana Flattop")
Ha.hist(a1_x3 , bins=20, density=True, alpha=0.7, edgecolor='black', label = "Estimador 1 ventana Blackmanharris")

Ha.set_title(f'Estimador de Amplitud (SNR = {SNR_db}dB)')
Ha.set_xlabel('Amplitud detectada')
Ha.legend()
Ha.grid(alpha=0.3)

# Hitograma freq 
Hf.hist(sigma1_x1 , bins=20, density=True, alpha=0.7, edgecolor='black', label = "Estimador 2 ventana unitaria")
Hf.hist(sigma1_x2 , bins=20, density=True, alpha=0.7, edgecolor='black', label = "Estimador 2 ventana Flattop")
Hf.hist(sigma1_x3 , bins=20, density=True, alpha=0.7, edgecolor='black', label = "Estimador 2 ventana Blackmanharris")

Hf.set_title(f'Estimador de Frecuencia (SNR = {SNR_db}dB)')
Hf.set_xlabel('Frecuencia detectada [Hz]')
Hf.legend()
Hf.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% ANALISIS

print(f"sumflat = {(np.sum(windows.flattop(N)))}")   
print(f"N = {N}") 
print(f"N/sumflat = {N/(np.sum(windows.flattop(N)))}")

print(f"N/sumBmh = {(np.sum(windows.blackmanharris(N)))}")   
print(f"N = {N}")
print(f"N/sumBmh = {N/(np.sum(windows.blackmanharris(N)))}")

# --- SESGOS Y VARIANZAS ---

# 1. Rectangular
# El estimador de amplitud (a1_x1) ya lo sacaste de mag_x1[int(N/4), :]
var_a1_x1 = np.var(a1_x1)
var_s1_x1 = np.var(sigma1_x1)
sesgo_a1_x1 = np.mean(a1_x1) - a0
sesgo_s1_x1 = np.mean(sigma1_x1 - f1)

# 2. Flat-top
# Usamos el estimador a1_x2 que extrajiste de mag_x2
var_a1_x2 = np.var(a1_x2)
var_s1_x2 = np.var(sigma1_x2)
sesgo_a1_x2 = np.mean(a1_x2) - a0
sesgo_s1_x2 = np.mean(sigma1_x2 -f1)

# 3. Blackman-Harris
# Usamos el estimador a1_x3 que extrajiste de mag_x3
var_a1_x3 = np.var(a1_x3)
var_s1_x3 = np.var(sigma1_x3)
sesgo_a1_x3 = np.mean(a1_x3) - a0
sesgo_s1_x3 = np.mean(sigma1_x3 - f1)

# Estructura para la tabla
resultados = [
    ["Rectangular", sesgo_a1_x1, var_a1_x1, sesgo_s1_x1, var_s1_x1],
    ["Flat-top",    sesgo_a1_x2, var_a1_x2, sesgo_s1_x2, var_s1_x2],
    ["Blackman-H",  sesgo_a1_x3, var_a1_x3, sesgo_s1_x3, var_s1_x3]
]

print(f"{'Ventana':<15} | {'Sesgo Amp':<10} | {'Var Amp':<10} | {'Sesgo Freq':<10} | {'Var Freq':<10}")
print("-" * 75)
for r in resultados:
    print(f"{r[0]:<15} | {r[1]:10.4f} | {r[2]:10.4f} | {r[3]:10.4f} | {r[4]:10.4f}")

