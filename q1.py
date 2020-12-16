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
    """
    INPUT:
        f: vector for frequencies to evaluate 
        phis: vector of [phi_1, .. phi_p] of AR(p) process
        sigma2: variance of the white noise process
    OUTPUT:
        S: spectral density fucntion of the AR(p) process evaluated at
        frequencies in f
    """
    p = len(phis) #Determine p
    A = 1 #initialise difference in the denominator
    for i in range(1,p+1):
        #Loop for each p to substract from denominator
        A = A - phis[i-1]*np.exp(-1j*2*np.pi*i*np.array(f)) 
    S = sigma2/(np.abs(A)**2) #compute spectral density function
    return S

def AR2_sim(phis,sigma2,N):
    """
    INPUT:
        phis: vector of [phi_1, phi_2] of Gaussian AR(2) process
        sigma2: variance of the white noise process
        N: desired length of output
    OUTPUT:
        X: vector of values of the generated AR(2) process, discarding 
        first 100 values
    """
    #generate white noise process
    et = np.random.normal(0, np.sqrt(sigma2), 100+N) 
    # initialise output X
    X = np.zeros(100+N)
    for t in range(2,100+N):
        #loop to define each element X_t
        X[t] = phis[0]*X[t-1]+phis[1]*X[t-2]+et[t]
    
    X = X[100:] #discard first 100 values
    
    return X

def acvs_hat(X,tau):
    """
    INPUT:
        X: vector of time series
        tau: vector of of values at which to estimate the autocovariance, 
        positive values
    OUTPUT:
        s: estimate of the autocovariance sequence at values in tau
    """
    #Define length of (X_1, ..., X_n)
    N = len(X)
    #initialise vector of autocovariance sequence
    s = np.zeros(len(tau))
    for i in range(len(tau)):
        #loop to define each s_tau
        s[i] = (1/N)*np.dot(X[:N-tau[i]], X[tau[i]:])
        
    return s

######################### QUESTION 2 #########################################

def periodogram(X):
    """
    INPUT:
        X: time series
    OUTPUT:
        S: periodogram of the time series at the Fourier frequencies
    """
    #Define length of the time series
    N = len(X)
    #Compute periodogram using fft
    S = (1/N)*np.abs(fft(X))**2
    
    return S

def direct(X):
    """
    INPUT:
        X: time series
    OUTPUT:
        S: direct spectral estimate of the time series at the Fourier 
        frequencies using the Hanning taper 
    """
    #Define length of the time series
    N = len(X)
    
    #Define the Hanning taper
    t = np.array(range(1,N+1))
    h = 0.5*(8/(3*(N+1)))**0.5 * (1-np.cos(2*np.pi*t/(N+1)))
    
    #Compute the direcy spectral estimate using the taper and fft
    S = np.abs(fft(np.multiply(h,X)))**2
    
    return S

def question2b():
    #vector of sample sizes to test
    N = [16, 32, 64, 128, 256, 512, 1024, 2048 , 4096]
    #[phi_1,2, phi_2,2,]
    phis = np.array([0.95*np.sqrt(2), -0.95**2])
    #Frequencies at which to evaluate the bias of estimates
    #true spectral density at these frequencies
    f = [1/8, 2/8, 3/8]
    S = S_AR(f, phis, 1)
    
    #initialise matrices containing bias for each frequency and sample size
    #and estimation method
    bias_p = np.zeros((3,len(N)))
    bias_d = np.zeros((3,len(N)))
    for j in range(len(N)):
        #loop for each sample size
        S_p = np.zeros((3,10000))
        S_d = np.zeros((3,10000))
        #generate time series
        X = AR2_sim(phis,1,N[j])
        for i in range(10000):
            #loop for 10000 realisations
            Sp = periodogram(X)
            Sd = direct(X)
            S_p[:,i] = np.array([Sp[2**(j+1)], Sp[2**(j+2)], Sp[6*2**j]]).T
            
            S_d[:,i] = np.array([Sd[2**(j+1)], Sd[2**(j+2)], Sd[6*2**j]]).T
 
        #calculate bias
        bias_p[:,j] = np.mean(S_p, axis = 1) - np.array(S).T
        bias_d[:,j] = np.mean(S_d, axis = 1) - np.array(S).T
    
    #plots
    plt.figure(figsize = (15,20))
    
    plt.subplot(311)
    plt.plot(np.log(N), bias_p[0,:], label = 'Periodogram')
    plt.plot(np.log(N), bias_d[0,:], label = 'Direct spectral estimate')
    plt.legend()
    plt.xlabel('Log N')
    plt.ylabel('Bias')
    plt.title('Bias of spectral estimators at frequency 1/8 for different' +
              ' values of N')
    
    plt.subplot(312)
    plt.plot(np.log(N), bias_p[1,:], label = 'Periodogram')
    plt.plot(np.log(N), bias_d[1,:], label = 'Direct spectral estimate')
    plt.legend()
    plt.xlabel('Log N')
    plt.ylabel('Bias')
    plt.title('Bias of spectral estimators at frequency 2/8 for different' +
              ' values of N')
    
    plt.subplot(313)
    plt.plot(np.log(N), bias_p[2,:], label = 'Periodogram')
    plt.plot(np.log(N), bias_d[2,:], label = 'Direct spectral estimate')
    plt.legend()
    plt.xlabel('Log N')
    plt.ylabel('Bias')
    plt.title('Bias of spectral estimators at frequency 3/8 for different' +
              ' values of N')
    
    plt.show()
    
    return None

def question2c():
    #Compute true spectral density
    phis = np.array([0.95*2**0.5, -0.95**2])
    f_l = [i/100 for i in range(100)]
    S = S_AR(f_l, phis, 1)
    #plots
    plt.plot(f_l,S)
    plt.vlines(1/8, 0, 160, color = 'r', linestyles = 'dashed', label = '1/8' )
    plt.vlines(2/8, 0, 160, color = 'g', linestyles = 'dashed', label = '2/8' )
    plt.vlines(3/8, 0, 160, color = 'b', linestyles = 'dashed', label = '3/8' )
    plt.legend()
    plt.xlabel('frequency f')
    plt.ylabel('S(f)')
    plt.title('Plot of the true spectral density function')
    plt.show()
    return None

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

def question3a(X):
    N = len(X)
    #compute periodogram with fftshift
    S_p = (1/N)*np.abs(fftshift(fft(X)))**2
    
    #Compute tapered direct spectral estimate with fftshift
    t = np.array(range(1,N+1))
    h = 0.5*(8/(3*(N+1)))**0.5 * (1-np.cos(2*np.pi*t/(N+1)))
    
    S_d = (1/N)*np.abs(fftshift(fft(np.multiply(h,X))))**2
    
    
    #Plots
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
    return None        

def yule_walker(X,p):
    """
    INPUT:
        X: time series of length 128
        p: order of AR(p) process that we want to fit to X
    OUTPUT:
        phi_hat = (big gamma)^-1 * small gamma, yule-walker estimator of AR(p)
        parameters phis
        sigma2_hat:  yule-walker estimator of variance of white noise process
    """
    s_hat = acvs_hat(X, list(range(0,p+1))) #estimate autocovariance sequence
    gamma = s_hat[1:] #define small gamma = [shat_1, ..., shat_p]
    #define big gamma, which is a symmetric Topelitx matrix
    Gamma = toeplitz(s_hat[:-1], s_hat[:-1]) 
    #compute phi_hat and sigma2_hat
    phi_hat = np.linalg.inv(Gamma).dot(gamma)
    sigma2_hat = s_hat[0] - np.dot(phi_hat,s_hat[1:])
    
    return phi_hat, sigma2_hat

def least_squares(X,p):
    """
    INPUT:
        X: time series of length 128
        p: order of AR(p) process that we want to fit to X
    OUTPUT:
        phis_hat = (F.T F)^-1 F.T X, least squares estimator of AR(p) parameters phis
        sigma2_hat:  lest squares estimator of variance of white noise process
    """
    N = len(X)
    #initiliase matrix F
    F = np.zeros((N-p,p))
    for i in range(p):
        # Loop to define each column of F
        F[:,i] = X[(p-i-1):(N-1-i)]
    #Compute phi_hat
    phi_hat = np.linalg.inv(F.T.dot(F)).dot(F.T).dot(X[p:])
    #Compute sigma2_hat
    sigma2_hat = (1/(N-2*p))*(X[p:]-F.dot(phi_hat)).T.dot(X[p:]-F.dot(phi_hat))
    
    return phi_hat, sigma2_hat

def approx_mle(X,p):
    """
    INPUT:
        X: time series of length 128
        p: order of AR(p) process that we want to fit to X
    OUTPUT:
        phis_hat: least squares estimator of AR(p) parameters phis (which is 
        equal to Approximate MLE)
        sigma2_hat:  approximate maximum likelihood estimator of sigma2, which
        is the lest squares estimator of sigma2 *(N-2p)/(N-p)
    """
    N = len(X)
    #Get least squares estimators of phi and sigma2
    phi_hat, sigma2_hat = least_squares(X, p)
    #Transform least squares estimator of sigma2 into approximate MLE
    sigma2_hat = sigma2_hat *(N-2*p)/(N-p)
    
    return phi_hat, sigma2_hat

def question3c(X):
    #Initiliase matrix of AICS
    AIC = np.zeros((20,3))
    N = len(X)
    for p in range(1,21):
        #Loop for computing AIC at each order p
        _, s_y = yule_walker(X, p)
        _, s_l = least_squares(X, p)
        _, s_m = approx_mle(X, p)
        sigma2s = np.array([s_y, s_l, s_m])
        ps = np.array([p, p, p])
        AIC[p-1,:] = 2*ps.T + N*np.log(sigma2s.T)
    
    #Summarise in a pandas DataFrame
    data_AIC = pd.DataFrame(data = AIC, 
                            index = ["p = "+ str(i) for i in range(1,21)],
                            columns = ['Yule-Walker', 
                                       'Least Squares', 'Approximate MLE'])
    
    return data_AIC

def question3d(X):
    #Compute fitted paramaters for chosen p
    py, sy = yule_walker(X,4)
    pl,sl = least_squares(X, 5)
    pm,sm = approx_mle(X, 5)
    
    #Summarise parameters into a pandas DataFrame
    data_params = pd.DataFrame(data = [[ py, pl, pm],
                                       [sy, sl, sm]],
                               index = ['phi_hat', 'sigma2_hat'],
                               columns = ['Yule-Walker', 
                                       'Least Squares', 'Approximate MLE'])
    return data_params

def question3e(X):
    #Compute fitted paramaters for chosen p
    py, sy = yule_walker(X,4)
    pl,sl = least_squares(X, 5)
    pm,sm = approx_mle(X, 5)
    
    #Compute spectral densities
    f = [i/128 for i in range(-64,65)]
    S_y = S_AR(f, py, sy)
    S_l = S_AR(f, pl, sl)
    S_m = S_AR(f, pm, sm)
    
    #plots
    fig, ax = plt.subplots(3, 1, sharex = True, sharey = True, figsize = (7,10))

    ax[0].plot(f, S_y, label = 'Yule-Walker')
    ax[0].set_title('Yule-Walker')
    ax[1].plot(f, S_l, label = 'Least Squares')
    ax[1].set_title('Least Squares')
    ax[2].plot(f, S_m, label = 'Approximate MLE')
    ax[2].set_title('Approximate MLE')
    fig.suptitle('Associated spectral density functions for each model',
                 size = 16)
    fig.text(0.5, 0.04, 'Frequencies f', va='center', ha='center')
    fig.text(0.04, 0.5, 'Spectral density', va='center', ha='center', 
             rotation='vertical')
    plt.show()
    
    return py, pl, pm

######################### QUESTION 4 #########################################

def question4(X):
    #Get fitted parameters
    py, pl, pm = question3e(X)
    
    #Create matrix to contain actual and forecast values
    X_f = np.zeros((128,4))
    X_f[:,0] = X
    X_f[:118,1] = X[:118]
    X_f[:118,2] = X[:118]
    X_f[:118,3] = X[:118]
    for i in range(10):
        #Loop to calculate forecast values for each method
        X_f[118 + i,1] = py[0]*X_f[118 + i-1,1] +py[1]*X_f[118 + i-2,1] \
            + py[2]*X_f[118 + i-3,1]+ py[3]*X_f[118 + i-4,1]
            
        X_f[118 + i,2] = pl[0]*X_f[118 + i-1,2] +pl[1]*X_f[118 + i-2,2] \
            + pl[2]*X_f[118 + i-3,2]+ pl[3]*X_f[118 + i-4,2] \
                + pl[4]*X_f[118 + i-5,2]
            
        X_f[118 + i,3] = pm[0]*X_f[118 + i-1,3] +pm[1]*X_f[118 + i-2,3] \
            + pm[2]*X_f[118 + i-3,3]+ pm[3]*X_f[118 + i-4,3] \
                + pm[4]*X_f[118 + i-5,3]
                
    #take values only from time point 110
    X_f = X_f[109:,:]
    
    #Plots
    t = [i for i in range(110, 129)]
    plt.figure(figsize = (10,7))
    plt.plot(t, X_f[:,0], label = 'True values', marker = 'o')
    plt.plot(t, X_f[:,1], label = 'Yule-Walker forecast', marker = 'o')
    plt.plot(t, X_f[:,2], label = 'Least Squares forecast', marker = 'o')
    plt.plot(t, X_f[:,3], label = 'Approcimate ML forecast', marker = 'o')
    plt.legend()
    plt.title('Plot of time series X at time points 110 to 128, with actual '+
              'values vs forecasted values from YS, LS and ML methods')
    plt.xlabel('Time point t')
    plt.ylabel('X_t / X_t(l)')
    plt.show()
    
    #summarise forecast and actual values into pandas Data Frame
    data_f = pd.DataFrame(X_f, index = ['t = '+str(i) for i in range(110,129)],
                          columns = ['Actual values', 'Yule-Walker', 
                                     'Least Squares', ' Approximate ML'])
    return data_f