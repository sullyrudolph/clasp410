#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 19:21:41 2023

@author: sully
"""

weight = input('Weight: ')
unit = input("(K)g or (L)bs: ")
if unit.upper() == "K":
    output_K = float(weight) * 2.204
    print("Weight in Lbs: " + str(output_K))
    print ("Nice")
if unit.upper() == "L":
    output_L = str(float(weight) * 0.4536)
    print("Weight in Kg: " + (output_L))
    print ("Nice")
else:
    print("Bruh fr no unit xD")
    print("Alright man")
