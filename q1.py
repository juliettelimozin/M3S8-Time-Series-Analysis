#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:20:25 2020

@author: juliette
"""
import numpy as np
from scipy.fft import fft

######################### QUESTION 1 #########################################

def S_AR(f,phis,sigma2):
    if str(type(phis)) != "<class 'numpy.ndarray'>" :
        raise ValueError('Please input a numpy array for phis')
    p = len(phis) #Determine p
    A = np.concatenate((np.array([1]), -phis), axis = 0)
    exps = np.exp(-1j*2*np.pi*np.asarray(range(0,p+1), dtype = 'complex'))
    S = [sigma2/np.dot(A, exps**i) for i in f]
    
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
    N = [32, 64, 128, 256, 512, 1000, 1024, 2048 , 4096]
    phis = np.array([2*0.95*np.cos(np.pi/4), -0.95**2])
    
    bias_p = np.zeros((3,len(N)))
    bias_d = np.zeros((3,len(N)))
    for j in range(len(N)):
        S_p = np.zeros((3,N[j]))
        S_d = np.zeros((3,N[j]))
        for i in range(N[j]):
            X = AR2_sim(phis,1,16)
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


        