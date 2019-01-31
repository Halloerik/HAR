#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle

import preprocess_data
import preprocessing_pamap2




def generate_pamap():
    dataset = os.path.join(os.path.expanduser('~'), 'Downloads/')
    target_filename = "/data/ealterma/pamap.dat"
    preprocessing_pamap2.generate_data(dataset, target_filename)

def generate_opportunity():
    dataset = os.path.join(os.path.expanduser('~'),
"Downloads/OpportunityUCIDataset.zip")
    target_filename = "/data/ealterma/opportunity_{}.dat"
    labels = ["gestures", "locomotion"]
    for label in labels:
        preprocess_data.generate_data(dataset, target_filename.format(label), label)   
    
    
    
#generate_pamap()
generate_opportunity()