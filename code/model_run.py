#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:14:57 2018

@author: Heqi
"""
import os

def model(input_dir,r):
    os.system("python main.py --data_test Demo " + "--dir_demo " + str(input_dir)+" --scale "+ str(r) +" --pre_train ../experiment/model/EDSR_baseline_x"+str(r)+".pt --test_only --save_results")  
    print("python main.py --data_test Demo " + "--dir_demo " + str(input_dir)+" --scale "+ str(r) +" --pre_train ../experiment/model/EDSR_baseline_x"+str(r)+".pt --test_only --save_results")
#input_dir = "../test"
#r = 2
#model(input_dir,r)