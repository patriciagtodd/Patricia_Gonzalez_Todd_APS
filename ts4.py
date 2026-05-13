
import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal


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
w1 = w1.reshape(1,N)
w1 = np.tile(w1, (R,1))

a1 = np.fft.fft(matriz_Sr,n=N,axis=0)/(N)  # normalizada con las muestras
#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
a1 = (np.abs(a1)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
a1_db = 10*np.log10(a1[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf

# Gráfico SOLO DEL DELTA
#plt.figure(figsize=(10, 10))
plt.plot(a1_db) 

# línea vertical en x = 3

plt.xlim(0, fs/2)
plt.legend()
plt.grid(0.01)
plt.show()
'''HASTA ACA SE QUE ESTOY ENTRE -2 Y 2'''
# %% VENTANA FLATTOP

# %%

w1 = (N)
w1 = w1.reshape(1,N)
w1 = np.tile(w1, (R,1))

a1 = np.fft.fft(matriz_S,n=N,axis=0)/(N)  # normalizada con las muestras
#n y axis lo que hacen es decirle en que direccion y como es cada grupo que quiero analizar
a1 = (np.abs(a1)**2)*2 
#me queda el espectro de un solo lado porque antes se distribuia 1/2 a cada lado

# paso a db
a1_db = 10*np.log10(a1[:N//2] + 1e-12) #db

ffreq = np.arange(0, N)*deltaf

# Gráfico SOLO DEL DELTA
#plt.figure(figsize=(10, 10))
plt.plot(a1_db) 

# línea vertical en x = 3

plt.xlim(0, fs/2)
plt.legend()
plt.grid(0.01)
plt.show()
'''HASTA ACA SE QUE ESTOY ENTRE -2 Y 2'''





# %% Estimadores


#para mi primer estimador lo que quiero es ver que hay en la feta de omega 0
estimador1_a1 = a1_db[int(N/4), :]


#quiero llegar a una matriz de rx4 de ahi hago el histograma, ese 4 son las 4 ventanas que quiero probar

'''
SEGUNDO ESTIMADOR -> AHORA TRABAJO EL ARGUMENTO
quiero quedarme con los maximos, osea todos los argumosntos N/4
aca llo que quiero es ver todos los maximos
hay lit una funcion que es argmax  y ese va a ser mi estimador (omega sombrerito),
 luego el sesgo es omega sombrerito menos omega
 quiero detectar esos maximos y por si acaso verificar que  esten bien
'''

