# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:46:23 2026

@author: Patog
"""

import numpy as np
import matplotlib.pyplot as plt

Y_a = lambda z: z**3 + z**2 + z + 1
X_a = lambda z: z**3

w = np.linspace(0, np.pi, 1000)

z = np.exp(1j*w)

H = Y_a(z) / X_a(z)

mod = np.abs(H)
fase = np.angle(H)



fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Magnitud
ax[0].plot(w, mod, color='C4', linewidth = 2)

ax[0].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[0].set_xlabel(r'$\omega$ [rad/muestra]')
ax[0].set_ylabel(r'$|H(e^{j\omega})|$')
ax[0].set_title('Respuesta en frecuencia Modulo')
ax[0].grid(True)

# Fase
ax[1].plot(w, fase, color='C9',linewidth = 2)

ax[1].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[1].set_xlabel(r'$\omega$ [rad/muestra]')
ax[1].set_ylabel('Fase [rad]')
ax[1].set_title('Respuesta en frecuencia Fase')
ax[1].grid(True)

fig.suptitle(r'$H(z)=1+z^{-1}+z^{-2}+z^{-3}$', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
plt.show()

# %% B

Y_b = lambda z: z**4 + z**3 + z**2 + z + 1
X_b = lambda z: z**4

w = np.linspace(0, np.pi, 1000)

z = np.exp(1j*w)

H = Y_b(z) / X_b(z)

mod = np.abs(H)

fase = np.angle(H)


fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Magnitud
ax[0].plot(w, mod, color='C4', linewidth = 2)

ax[0].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[0].set_xlabel(r'$\omega$ [rad/muestra]')
ax[0].set_ylabel(r'$|H(e^{j\omega})|$')
ax[0].set_title('Respuesta en frecuencia Modulo')
ax[0].grid(True)

# Fase
ax[1].plot(w, fase, color='C9', linewidth = 2)

ax[1].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[1].set_xlabel(r'$\omega$ [rad/muestra]')
ax[1].set_ylabel('Fase [rad]')
ax[1].set_title('Respuesta en frecuencia Fase')
ax[1].grid(True)

fig.suptitle(r'$H(z)=1+z^{-1}+z^{-2}+z^{-3}+z^{-4}$', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
plt.show()

# %%

Y_c = lambda z:  z - 1
X_c = lambda z: z

w = np.linspace(0, np.pi, 1000)

z = np.exp(1j*w)

H = Y_c(z) / X_c(z)

mod = np.abs(H)

fase = np.angle(H)


fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Magnitud
ax[0].plot(w, mod, color='C4', linewidth = 2)

ax[0].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[0].set_xlabel(r'$\omega$ [rad/muestra]')
ax[0].set_ylabel(r'$|H(e^{j\omega})|$')
ax[0].set_title('Respuesta en frecuencia Modulo')
ax[0].grid(True)

# Fase
ax[1].plot(w, fase, color='C9', linewidth = 2)

ax[1].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[1].set_xlabel(r'$\omega$ [rad/muestra]')
ax[1].set_ylabel('Fase [rad]')
ax[1].set_title('Respuesta en frecuencia Fase')
ax[1].grid(True)

fig.suptitle(r'$H(z)= 1 - z^{-1}', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
plt.show()

# %% d

Y_d = lambda z:  z**2 - 1
X_d = lambda z: z**2

w = np.linspace(0, np.pi, 1000)

z = np.exp(1j*w)

H = Y_d(z) / X_d(z)

mod = np.abs(H)

fase = np.angle(H)


fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Magnitud
ax[0].plot(w, mod, color='C4', linewidth = 2)

ax[0].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[0].set_xlabel(r'$\omega$ [rad/muestra]')
ax[0].set_ylabel(r'$|H(e^{j\omega})|$')
ax[0].set_title('Respuesta en frecuencia Modulo')
ax[0].grid(True)

# Fase
ax[1].plot(w, fase, color='C9', linewidth = 2)

ax[1].set_xticks(
    [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
    ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
)

ax[1].set_xlabel(r'$\omega$ [rad/muestra]')
ax[1].set_ylabel('Fase [rad]')
ax[1].set_title('Respuesta en frecuencia Fase')
ax[1].grid(True)

fig.suptitle(r'$H(z)= 1 - z^{-2}', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.tight_layout()
plt.show()

