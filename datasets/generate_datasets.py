#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

import preprocess_data
import preprocessing_pamap2




def generate_pamap():
    dataset = r"C:/Users/Erik/Desktop/Universität Folien/Bachelor arbeit/"
    target_filename = "pamap.dat"
    preprocessing_pamap2.generate_data(dataset, target_filename)

def generate_opportunity():
    dataset = r"C:\Users\Erik\Desktop\Universität Folien\Bachelor arbeit\OpportunityUCIDataset.zip"
    target_filename = "opportunity_{}.dat"
    labels = ["gestures", "locomotion"]
    for label in labels:
        preprocess_data.generate_data(dataset, target_filename.format(label), label)   
    
    
    
#generate_pamap()
generate_opportunity()