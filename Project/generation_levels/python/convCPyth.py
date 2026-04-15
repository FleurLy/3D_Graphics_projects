#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
using Ctypes to access C mipmap function
"""
import ctypes
import os
import sys
from numpy.ctypeslib import ndpointer

def filtrePyth():
    ext = 'dll' if sys.platform.startswith('win') else 'so'
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'so', f'filtre.{ext}')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"La librairie compilée introuvable: {lib_path}. Veuillez compiler le code C.")
        
    dll = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.FGP_TV
    func.restype = None
    
    func.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),#input
                ctypes.c_float, # lambda
                ctypes.c_int, # iterations 
                ctypes.c_float, # epsil - tolerance
                ctypes.c_int, # methTV = 0 - isotropic, 1- anisotropic
                ctypes.c_int, # nonegativity = 0 (off)
                ctypes.c_int, # printing =0 (off)
                ctypes.c_int, # dimX
                ctypes.c_int, # dimY
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")] # output
    return func


def moyPyth():
    ext = 'dll' if sys.platform.startswith('win') else 'so'
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'so', f'moy.{ext}')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"La librairie compilée introuvable: {lib_path}. Veuillez compiler le code C.")
        
    dll = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.moy
    func.restype = None
    
    func.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),#input
                ctypes.c_int, # maskSize
                ctypes.c_int, # dimX
                ctypes.c_int, # dimY
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # output
                ctypes.c_int] # padMode
    return func


def medPyth():
    ext = 'dll' if sys.platform.startswith('win') else 'so'
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'so', f'med.{ext}')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"La librairie compilée introuvable: {lib_path}. Veuillez compiler le code C.")
        
    dll = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.med
    func.restype = None
    
    func.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),#input
                ctypes.c_int, # maskSize
                ctypes.c_int, # dimX
                ctypes.c_int, # dimY
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # output
                ctypes.c_int] # padMode
    return func

def miNePyth():
    ext = 'dll' if sys.platform.startswith('win') else 'so'
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'so', f'miNe.{ext}')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"La librairie compilée introuvable: {lib_path}. Veuillez compiler le code C.")
        
    dll = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.miNe
    func.restype = None
    
    func.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),#input
                ctypes.c_int, # dimX
                ctypes.c_int, # dimY
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # output
                ctypes.c_int] # padMode
    return func

def KaiserPyth():
    ext = 'dll' if sys.platform.startswith('win') else 'so'
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'so', f'kaiser.{ext}')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"La librairie compilée introuvable: {lib_path}. Veuillez compiler le code C.")
        
    dll = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    func = dll.kaiser
    func.restype = None
    
    func.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),#input
                ctypes.c_int, # maskSize
                ctypes.c_int, # dimX
                ctypes.c_int, # dimY
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # output
                ctypes.c_int,# padMode
                ctypes.c_float] #alpha
    return func
