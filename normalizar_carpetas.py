#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:08:39 2022

@author: pablo
"""
import os
import random
import shutil

categoria = 'DESODO_UNI'
direccion = '/home/ago/ramdisk/DS_'+categoria+'/'
UPCS = os.listdir(direccion+categoria+'_train')
L = []
U= len(UPCS)
Carpetas_creadas = []
for UPC in UPCS:
        L.append(len(os.listdir(direccion+categoria+'_train/'+UPC)))
        if os.path.exists(direccion+categoria+'_val/'+UPC) == False :
            os.mkdir(direccion+categoria+'_val/'+UPC)
            Carpetas_creadas.append(UPC)

minimo = int(min(L)*0.9) #100

#90-10 



normaliza_val_min = int(minimo * (15/100)) 
normaliza_val_max = int(minimo * (20/100)) 
for UPC in UPCS:
    rafagas = os.listdir(direccion+categoria+'_train/'+UPC)
    random_rafagas = random.sample(rafagas, k = minimo)
    N = len(os.listdir(direccion+categoria+'_train/'+UPC)) - minimo
    l_val = len(os.listdir(direccion+categoria+'_val/'+UPC))
    if normaliza_val_min - l_val > 0:
        limite = min([N,normaliza_val_min - l_val])
        count = 0
        for rafaga in rafagas:
            if rafaga not in random_rafagas:
                if count < limite:
                    shutil.copy(direccion+categoria+'_train/'+UPC+'/'+rafaga, direccion+categoria+'_val/'+UPC+'/'+rafaga)
                    count+=1
                os.remove(direccion+categoria+'_train/'+UPC+'/'+rafaga)
    elif l_val - normaliza_val_max > 0:
        random_val = random.sample(os.listdir(direccion+categoria+'_val/'+UPC), k = normaliza_val_max)
        for validation in os.listdir(direccion+categoria+'_val/'+UPC):
            if validation not in random_val:
                os.remove(direccion+categoria+'_val/'+UPC+'/'+validation)    
        for rafaga in rafagas:
            if rafaga not in random_rafagas:
                os.remove(direccion+categoria+'_train/'+UPC+'/'+rafaga)
    else:
        for rafaga in rafagas:
            if rafaga not in random_rafagas:
                os.remove(direccion+categoria+'_train/'+UPC+'/'+rafaga)
        
    
UPCS = os.listdir(direccion+categoria+'_train')
for UPC in UPCS:
        print(len(os.listdir(direccion+categoria+'_train/'+UPC)))
        print(len(os.listdir(direccion+categoria+'_val/'+UPC))) 
