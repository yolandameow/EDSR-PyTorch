#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:14:57 2018

@author: Heqi
"""
import os

def model(input_dir,r):
    os.system("main.py --data_test Demo --scale "+r +"--pre_train ../experiment/model/EDSR_baseline_x"+r+".pt --test_only --save_results"+"--dir_demo" + input_dir)
    
input_dir = "../cityscape_input"
r = 2
model(input_dir,r)