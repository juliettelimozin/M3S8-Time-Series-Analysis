#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:20:25 2020

@author: juliette
"""
import numpy as np
from scipy.fft import fft, fftshift
from scipy.linalg import toeplitz
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 0)
######################### QUESTION 1 #########################################

def S_AR(f,phis,sigma2):
    p = len(phis) #Determine p
    A = 1
    for i in range(1,p+1):
        A = A - phis[i-1]*np.exp(-1j*2*np.pi*np.array(f))
    S = sigma2/np.abs(A)**2
    return S

def AR2_sim(phis,sigma2,N):
    et = np.random.normal(0, np.sqrt(sigma2), 100+N)
    X = np.zeros(100+N)
    for t in range(2,100+N):
        X[t] = phis[0]*X[t-1]+phis[1]*X[t-2]+et[t]
    
    X = X[100:]
    
    return X

def acvs_hat(X,tau):
    N = len(X)
    s = np.zeros(len(tau))
    for i in range(len(tau)):
        s[i] = (1/N)*np.dot(X[:N-tau[i]], X[tau[i]:])
    
    return s

######################### QUESTION 2 #########################################

def periodogram(X):
    N = len(X)
    S = (1/N)*np.abs(fft(X))**2
    
    return S

def direct(X):
    N = len(X)
    t = np.array(range(1,N+1))
    h = 0.5*(8/(3*(N+1)))**0.5 * (1-np.cos(2*np.pi*t/(N+1)))
    
    S = (1/N)*np.abs(fft(np.multiply(h,X)))**2
    
    return S

def question2b():
    N = [16, 32, 64, 128, 256, 512, 1024, 2048 , 4096]
    phis = np.array([2*0.95*np.cos(np.pi/4), -0.95**2])
    
    bias_p = np.zeros((3,len(N)))
    bias_d = np.zeros((3,len(N)))
    for j in range(len(N)):
        S_p = np.zeros((3,10000))
        S_d = np.zeros((3,10000))
        for i in range(10000):
            X = AR2_sim(phis,1,N[j])
            Sp = periodogram(X)
            Sd = direct(X)
            S_p[0,i] = Sp[2]
            S_p[1,i] = Sp[4]
            S_p[2,i] = Sp[6]
            S_d[0,i] = Sd[2]
            S_d[1,i] = Sd[4]
            S_d[2,i] = Sd[6]
            
        f = [1/8, 2/8, 3/8]
        S = S_AR(f, phis, 1)
        for i in range(3):
            bias_p[i,j] = np.mean(S_p[i,:])-S[i]
            bias_d[i,j] = np.mean(S_d[i,:])-S[i]
            
    print(bias_p)
    print(bias_d)

######################### QUESTION 3 #########################################

X = [ -1.817,1.2148,0.11579,-1.1119,-0.57995,1.4834,-0.24767,0.32744,-1.1843,
     -0.53556,2.0333,0.083949,-1.2065,1.0027,-0.54753,0.94977,-2.8038,2.4326,
     0.038378,-0.23335,-1.8316,0.88818,2.1609,-3.4387,1.3854,-2.2549,5.6756,
     -7.2475,4.3075,-2.1435,3.3226,-2.6033,0.527,-1.4745,3.652,-4.1965,3.1151,
     -3.5564,5.1222,-3.6464,0.33439,0.30982,0.039963,0.76926,-1.5944,3.5179,
     -4.3289,2.7233,-1.4127,1.9421,-0.64151,-0.45006,-1.0018,1.9401,0.3872,
     -2.5166,1.2582,-0.25959,1.492,-1.354,0.98541,-1.5397,2.8433,-3.0394
     ,2.6169,-2.1013,1.9765,-0.95334,-1.3135,3.7725,-4.1135,4.5512,-6.129,
     8.0767,-8.7753,6.6041,-3.7521,1.9787,-1.2391,0.0066444,1.4772,0.25713,
     -2.7869,1.8693,-0.6165,1.6516,-2.9129,3.2405,-2.3272,1.8146,-2.8721,
     1.8328,-0.015361,-0.15966,-1.1049,0.4769,0.18141,-0.11041,1.3148,-3.6286,
     3.0333,-1.0317,0.89456,-1.6652,2.304,-2.6236,2.3357,-2.9236,4.154,-3.7316,
     2.1474,-1.6903,2.2387,-2.2171,1.7293,-1.1878,0.26944,1.7637,-3.499,3.6687
     ,-4.2159,4.401,-3.8027,2.5042,0.3691,-2.3709,2.9599,-3.2685]
N = len(X)
S_p = (1/N)*np.abs(fftshift(fft(X)))**2

t = np.array(range(1,N+1))
h = 0.5*(8/(3*(N+1)))**0.5 * (1-np.cos(2*np.pi*t/(N+1)))

S_d = (1/N)*np.abs(fftshift(fft(np.multiply(h,X))))**2

f = [i/128 for i in range(-64,64)]
plt.figure(figsize = (20,10))
plt.subplot(121)
plt.plot(f, S_p)
plt.xlabel('frequencies f')
plt.ylabel('Periodogram')
plt.title('Periodogram of my time series at frequencies [-0.5,0.5]')

plt.subplot(122)
plt.plot(f, S_d)
plt.xlabel('frequencies f')
plt.ylabel('Direct spectral estimate')
plt.title('Direct spectral estimate of my time series using the Hanning taper'
          +' at frequencies [-0.5,0.5]')
plt.show()         

def yule_walker(X,p):
    s_hat = acvs_hat(X, list(range(0,p+1)))
    gamma = s_hat[1:]
    Gamma = toeplitz(s_hat[:-1], s_hat[:-1])
    phi_hat = np.linalg.inv(Gamma).dot(gamma)
    sigma2_hat = s_hat[0] - np.dot(phi_hat,s_hat[1:])
    
    return phi_hat, sigma2_hat

def least_squares(X,p):
    N = len(X)
    F = np.zeros((N-p,p))
    for i in range(p):
        F[:,i] = X[(p-i-1):(N-1-i)]
    phi_hat = np.linalg.inv(F.T.dot(F)).dot(F.T).dot(X[p:])
    sigma2_hat = (1/(N-2*p))*(X[p:]-F.dot(phi_hat)).T.dot(X[p:]-F.dot(phi_hat))
    
    return phi_hat, sigma2_hat

def approx_mle(X,p):
    N = len(X)
    phi_hat, sigma2_hat = least_squares(X, p)
    sigma2_hat = sigma2_hat *(N-2*p)/(N-p)
    
    return phi_hat, sigma2_hat

AIC = np.zeros((20,3))
N = len(X)
for p in range(1,21):
    _, s_y = yule_walker(X, p)
    _, s_l = least_squares(X, p)
    _, s_m = approx_mle(X, p)
    sigma2s = np.array([s_y, s_l, s_m])
    ps = np.array([p, p, p])
    AIC[p-1,:] = 2*ps.T + N*np.log(sigma2s.T)
    
data_AIC = pd.DataFrame(data = AIC, 
                        index = ["p = "+ str(i) for i in range(1,21)],
                        columns = ['Yule-Walker', 
                                   'Least Squares', 'Approximate MLE'])

py, sy = yule_walker(X,4)
pl,sl = least_squares(X, 5)
pm,sm = approx_mle(X, 5)

data_params = pd.DataFrame(data = [[ py, pl, pm],
                                   [sy, sl, sm]],
                           index = ['phi_hat', 'sigma2_hat'],
                           columns = ['Yule-Walker', 
                                   'Least Squares', 'Approximate MLE'])


f = [i/128 for i in range(-64,65)]
S_y = S_AR(f, py, sy)
S_l = S_AR(f, pl, sl)
S_m = S_AR(f, pm, sm)
plt.plot(f, S_y, label = 'Yule-Walker')
plt.plot(f, S_l, label = 'Least Squares')
plt.plot(f, S_m, label = 'Approximate MLE')
plt.legend()
plt.xlabel('frequencies f')
plt.ylabel('Spectral density function S(f)')
plt.title('Associated spectral density functions for each model')
plt.show()

    