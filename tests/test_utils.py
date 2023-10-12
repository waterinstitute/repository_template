""" Tests for functions in src/data/utils.py """
#  export PYTHONPATH="/home"
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime
import pytest 
import os
from pathlib import Path

import sys

from src.data.utils import *

class TestGetHUC12():
    @pytest.mark.parametrize("point,huc", [
        ((-91.69396, 41.89110), "070802081007"),
        ((-87.8834,42.0334), "071200040505") 
    ])
    def test_get_huc12_valid_point(self, point, huc):
        huc12_returned = get_huc12(point)
        assert huc12_returned == huc
        
    def test_get_huc12_invalid_point(self):
        with pytest.raises(Exception):
            get_huc12((100,200))
            
            
class TestMaskArray():
    
    @pytest.mark.parametrize("arr_to_mask", [
        (np.array([1.0,2.0,3.0,4.0,5.0])),
        (np.array([0.0,0.0,0.0,0.0,0.0]))
    ])
    def test_mask_array(self, arr_to_mask):
        mask_arr = np.array([1,0,0,0,1])
        result = mask_array(mask_arr, arr_to_mask)
        
        # loop through for comparisons since nan != nan
        for i in range(len(result)):
            if mask_arr[i] == 0:
                assert np.isnan(result[i])
            else:
                assert arr_to_mask[i] == result[i]

    @pytest.mark.parametrize("arr_to_mask", [
        (np.array([1,2,3,4,5])),
        (np.array([0,0,0,0,0]))
    ])
    def test_mask_array_int(self,arr_to_mask):
        mask_arr = np.array([1,0,0,0,1])
        result = mask_array(mask_arr, arr_to_mask)
        
        # loop through for comparisons since nan != nan
        for i in range(len(result)):
            if mask_arr[i] == 0:
                assert np.isnan(result[i])
            else:
                assert arr_to_mask[i] == result[i]
            
    def test_mask_array_diff_dim(self):
        mask_arr = np.array([1,0,0,0,1,0,0,0])
        arr_to_mask = np.array([1,2,3,4,5])
        with pytest.raises(Exception):
            result = mask_array(mask_arr, arr_to_mask)