
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

r = mi_funcion_ruido(N, Pr)
r = r.reshape(N,1)
r = np.tile(r, (1,R))

matriz_Sr = matriz_S + r


# %% SIN VENTANA

w1 = np.ones(N)
w1 = w1.reshape(N,1)
w1 = np.tile(w1, (1,R))

a1 = np.fft.fft(matriz_Sr,n=N,axis=0)/(N)  # normalizada con las muestras
#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
a1 = (np.abs(a1)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
a1_db = 10*np.log10(a1[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf


plt.plot(a1_db) 

plt.xlim(0, fs/2)
plt.legend()
plt.grid(0.01)
plt.show()
'''HASTA ACA SE QUE ESTOY ENTRE -2 Y 2'''


# %% VENTANA FLATTOP

w2 = windows.flattop(N)
w2 = w2.reshape(N,1)
w2 = np.tile(w2, (1,R))

a2 = np.fft.fft((matriz_Sr*w2),n=N,axis=0)/(N)  # normalizada con las muestras
#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
a2 = (np.abs(a2)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
a2_db = 10*np.log10(a2[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf

# Gráfico SOLO DEL DELTA
#plt.figure(figsize=(10, 10))
plt.plot(a2_db) 

# línea vertical en x = 3

plt.xlim(0, fs/2)
plt.legend()
plt.grid(0.01)
plt.show()
'''HASTA ACA SE QUE ESTOY ENTRE -2 Y 2'''



# %% VENTANA Blackmanharris

w3 = windows.blackmanharris(N)
w3 = w3.reshape(N,1)
w3 = np.tile(w3, (1,R))

a3 = np.fft.fft((matriz_Sr*w3),n=N,axis=0)/(N)  # normalizada con las muestras
#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
a3 = (np.abs(a3)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
a3_db = 10*np.log10(a3[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf

# Gráfico SOLO DEL DELTA
#plt.figure(figsize=(10, 10))
plt.plot(a3_db) 

# línea vertical en x = 3

plt.xlim(0, fs/2)
plt.legend()
plt.grid(0.01)
plt.show()
'''HASTA ACA SE QUE ESTOY ENTRE -2 Y 2'''

# %% Estimadores

#para mi primer estimador lo que quiero es ver que hay en la feta de omega 0
estimador1_a1 = a1[int(N/4), :]
estimador1_a2 = a2[int(N/4), :]
estimador1_a3 = a3[int(N/4), :]

#quiero llegar a una matriz de rx4 de ahi hago el histograma, ese 4 son las 4 ventanas que quiero probar

'''
SEGUNDO ESTIMADOR -> AHORA TRABAJO EL ARGUMENTO
quiero quedarme con los maximos, osea todos los argumosntos N/4
aca llo que quiero es ver todos los maximos
hay lit una funcion que es argmax  y ese va a ser mi estimador (omega sombrerito),
 luego el sesgo es omega sombrerito menos omega
 quiero detectar esos maximos y por si acaso verificar que  esten bien
'''
estimador2_a1 = np.argmax(a1, axis = 0)
estimador2_a2 = np.argmax(a2, axis = 0)
estimador2_a3 = np.argmax(a3, axis = 0)

Mu_a = np.mean(a1)

sesgo = Mu_a - a0

#%% Hitograma amplitud 
plt.hist(estimador1_a1, bins=15, density=True, alpha=0.7, edgecolor='black', label = "Estimador 1 ventana unitaria")
plt.hist(estimador1_a2, bins=15, density=True, alpha=0.7, edgecolor='black', label = "Estimador 1 ventana Flattop")
plt.hist(estimador1_a3, bins=15, density=True, alpha=0.7, edgecolor='black', label = "Estimador 1 ventana Blackmanharris")

# Hitograma arg 
plt.hist(estimador2_a1, bins=15, density=True, alpha=0.7, edgecolor='black', label = "Estimador 2 ventana unitaria")
plt.hist(estimador2_a2, bins=15, density=True, alpha=0.7, edgecolor='black', label = "Estimador 2 ventana Flattop")
plt.hist(estimador2_a3, bins=15, density=True, alpha=0.7, edgecolor='black', label = "Estimador 2 ventana Blackmanharris")

plt.grid(True)
plt.legend()
plt.show()




