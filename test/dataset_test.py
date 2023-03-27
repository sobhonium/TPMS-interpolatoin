
# to run the test on terminal
# >>  python -m unittest dataset_test.py 
# or
# >> python dataset_test.py

import unittest
import numpy as np
import torch

import sys
sys.path.insert(0,"..")
from dataset.tig_based_dataset import load_gyroid_sdf_dataset



class CheckTestClass(unittest.TestCase):
    '''destined to test functionalities and utilities'''
    

    def setUp (self):
        coef=list([[1,2,1]])
        self.features, self.labels = load_gyroid_sdf_dataset(coef=coef, axis_chuncks=10)
        
        

    # will be checked since it has test_ at the begingin.
    def test_gyroid_dataset(self):

        # dataset must be of type torch.tensor 
        actual_features, actual_labels = self.features, self.labels 
        expected_features_size =  torch.Size([1000, 6])
        expected_labels_size   =  torch.Size([1000])
        self.assertEqual(actual_features.shape,expected_features_size)
        self.assertEqual(actual_labels.shape,expected_labels_size)
        
    



if __name__ == '__main__':
    # using and running unittest.main() let us 
    # run this file without any other options or 
    # arguments. Otherwise, we need to run
    # >> python -m unittest <file_name>.py
    # each time we want to test.
    unittest.main()