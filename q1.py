#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:20:25 2020

@author: juliette
"""
import numpy as np

def S_AR(f,phis,sigma2):
    """
    Write a function S_AR(f,phis,sigma2) that evaluates the parametric 
    form of the spectral density function for an AR(p) process on a designated 
    set of frequencies.
    The inputs should be:
    f: a vector of frequencies at which to evaluate the spectral density 
    function.
    phis: the vector [φ1,p, ..., φp,p].
    sigma2: the variance of the white noise.
    Your function should not take p as an input, but instead determine 
    it from the stated input
    parameters.
    It should return a single output:
    S: a vector of values of the spectral density function evaluated at the 
    elements of f.
    """
    if str(type(phis)) != "<class 'numpy.ndarray'>" :
        raise ValueError('Please input a numpy array for phis')
    p = len(phis) #Determine p
    A = np.concatenate((np.array([1]), -phis), axis = 0)
    exps = np.exp(-1j*2*np.pi*np.asarray(range(0,p+1), dtype = 'complex'))
    S = [sigma2/np.dot(A, exps**i) for i in f]
    
    return S


        