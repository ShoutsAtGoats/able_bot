import numpy as np
import matplotlib.pyplot as plt


def modulation_shape(repeat=1):
    """ Function which shows a modulation shape """
    t = np.arange(0, 80, 0.1)
    amplitude = np.cos(t)
    plt.plot(t[1:60], amplitude[1:60])
    plt.title("Modulation Shape")
    plt.xlabel('Time')
    plt.ylabel('Amplitude = sin(time)')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.show()

modulation_shape()



