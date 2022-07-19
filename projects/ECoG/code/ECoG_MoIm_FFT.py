#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:34:54 2022

@author: gaby, ruoqi
"""
#General info about sine waves:

#Sine waves represent periodic oscillations.
#Sine waves have the shape of sine curve.
#The X-axis of the sine curve represents the time.
#The Y-axis of the sine curve represents the amplitude of the sine wave.
#The amplitude of the sine wave at any point in Y is proportional to the sine of a variable.
#The sine curve goes through origin.
#A cycle of sine wave is complete when the position of the sine wave starts from a position and comes to the same position after attaining its maximum and minimum amplitude during its course.
#The time taken to complete one cycle is called the period of the sine wave.
#The frequency of the sine wave is given by number of cycles per second.
#The distance covered by a cycle measures the wavelength of the sine wave (λ).
#The sine wave is given by the equation: A sin(ωt)
#A-Amplitude; t-Time; ω(Omega)-Frequency

#The following code is an example of how to graph a sine wave using numpy and matplotlib:
#Thanks for providing the fantastic starting point Gabby!


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.fft import fft, ifft
from scipy import signal


# In[7]:


#adapted from Gabby's work, written in functions
def sine_generator(freq, phase, Sampling_Rate):
    samplingInterval = 1/Sampling_Rate
    t = np.arange(0,1,samplingInterval)
    x_1 = np.sin(np.pi*2*freq*t+phase)
    return x_1

def fft_func(sine_wave):
    X = fft(x_1)
    N = len(X)
    n = np.arange(N)
    T = N/samplingRate
    freq = n/T
    return freq
    
def plot_fft(sine_wave):
    freq = fft_func(sine_wave)
    amplitude = np.abs(X)
    plt.stem(freq, amplitude, 'b')

    plt.xlabel('Freq (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 10)
    plt.title('FFT of sine wave_1') 

def plot_sine_1(samplingRate, freq, phase):
    #samplingRate = 200

    # sampling interval
    samplingInterval = 1/samplingRate
    t = np.arange(0,1,samplingInterval)

    #set frequency
    #freq = 8
    #phase = 3.1415/2
    x = np.sin(np.pi*2*freq*t+phase)
    #plot wave, line color = blue
    sine_plot = plt.plot(t, x, 'b')
    return sine_plot


# In[8]:


wave_1 = sine_generator(4, 0, 200)
plot_fft(wave_1)
plot_sine_1(200, 4, 0)


# In[90]:


#power
#From https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
from scipy import signal
import matplotlib.pyplot as plt
rng = np.random.default_rng()
def power_visualizer(signal, noise_power, fs):
    #get the frequency from fast fourier transformation
    X = fft(x_1)
    N = len(X)
    n = np.arange(N)
    T = N/samplingRate
    freq = n/T
    #amplitude may be going wrong here
    amp = np.abs(X)
    frq = fft_func(signal)
    
    rng = np.random.default_rng()
    time = np.arange(N) / fs
    x = amp*np.sin(2*np.pi*frq*time)
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    f, Pxx_den = scipy.signal.periodogram(x, fs)
    #plot the power
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    
    #power specturm
    
    f, Pxx_spec = scipy.signal.periodogram(x, fs, 'flattop', scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.ylim([1e-4, 1e1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.show()
    
    noise_rec = np.mean(Pxx_den[100:])
    peak = np.sqrt(Pxx_spec.max())
    return noise_rec, peak


# In[91]:


#power_visualizer(wave, 0.001 * 10e3 / 2, 10e3)
wave = sine_generator(1234.0, 3.5, 2000)
power_visualizer(wave, 0.001 * 10e3 / 2, 10e3)


# In[7]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:34:54 2022

@author: gaby
"""
#General info about sine waves:

#Sine waves represent periodic oscillations.
#Sine waves have the shape of sine curve.
#The X-axis of the sine curve represents the time.
#The Y-axis of the sine curve represents the amplitude of the sine wave.
#The amplitude of the sine wave at any point in Y is proportional to the sine of a variable.
#The sine curve goes through origin.
#A cycle of sine wave is complete when the position of the sine wave starts from a position and comes to the same position after attaining its maximum and minimum amplitude during its course.
#The time taken to complete one cycle is called the period of the sine wave.
#The frequency of the sine wave is given by number of cycles per second.
#The distance covered by a cycle measures the wavelength of the sine wave (λ).
#The sine wave is given by the equation: A sin(ωt)
#A-Amplitude; t-Time; ω(Omega)-Frequency



#The following code is an example of how to graph a sine wave using numpy and matplotlib:

# importing NumPy
import numpy as np


# importing matplotlib
import matplotlib.pyplot as plt

# sampling rate
samplingRate = 200

# sampling interval
samplingInterval = 1/samplingRate
t = np.arange(0,1,samplingInterval)

#set frequency
freq = 4
phase = 0
x = np.sin(np.pi*2*freq*t+phase)


#plot wave, line color = blue
plt.plot(t, x, '#cf3a8e')

sine=plt.gca()
# Give a title for the sine wave plot
#plt.title('Sample sine wave')


# Give x axis label for the sine wave plot
plt.xlabel('Time (s)', fontsize=30)

# Give y axis label for the sine wave plot
plt.ylabel('Neural Activity', fontsize=30)

# add gridlines to plot
plt.grid(False)

sine.axes.yaxis.set_ticklabels([])
plt.xticks(fontsize= 16 ) 
 
# add line at y=0, red in color
plt.axvline(x=0.25, color='grey', linestyle="dotted")
plt.axvline(x=0.5, color='grey', linestyle="dotted")
plt.axvline(x=0.75, color='grey', linestyle="dotted")


#plt.show()





#---------------------------------------------------------------------------------------------------------------


# In[ ]:



# Give a title for the sine wave plot
plt.title('Sample sine wave')


# Give x axis label for the sine wave plot
plt.xlabel('Time')

# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = sin(time)')

# add gridlines to plot
plt.grid(True)

 
# add line at y=0, red in color
plt.axhline(y=0, color='r')


plt.show()


# In[98]:


#The following code is an example of how to graph a FFT in python using numpy and matplotlib:

from numpy.fft import fft, ifft

# sampling rate


plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 10)

# Give a title for the sine wave plot
plt.title('FFT of sine wave')


# In[119]:


def fft_plot(samplingRate, freq):
    #samplingRate = 100
    #sampling interval
    samplingInterval = 1/samplingRate
    t = np.arange(0,1,samplingInterval)
    #set frequency
    freq = 8.
    x = np.sin(np.pi*2*freq*t)
    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/samplingRate
    freq = n/T 
    samp_lim = samplingRate/2
    fft_plot = plt.stem(freq, np.abs(X), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0,samp_lim)

    # Give a title for the sine wave plot
    plt.title('FFT of sine wave')

    return fft_plot


# In[120]:


fft_plot(100, 8)


# In[ ]:


#new analysis

