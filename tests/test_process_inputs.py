""" Tests for functions in src/data/process_inputs.py """
#  export PYTHONPATH="/home"
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime
import pytest 
import os
from pathlib import Path

import sys

from src.data.process_inputs import *

DATA_PATH = "/home/jovyan/app/data"
HUC12_PATH = str(Path(DATA_PATH, "huc12"))
SRC_PATH = "/home/src"

def test_check_file_exist():
    """ Test that check_files_exist works on list of one file"""
    for file in os.listdir(SRC_PATH):
        assert check_files_exist([Path(SRC_PATH, file)])
        
def test_check_files_exist():
    """ Test that check_files_exist works on list of multiple files """
    assert check_files_exist([Path(SRC_PATH, file) for file in os.listdir(SRC_PATH)])

@pytest.mark.parametrize("d", [
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
     'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
     'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
     'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]}
])
class TestWriteFile:
    
    def test_write_file_parquet(self, d):
        """ test write_file can write to parquet """
        df = pd.DataFrame(d)
        write_file(df, "test.parquet")
        os.remove("test.parquet") # clean up
    
    def test_write_file_geojson(self, d):
        """ test write_file can write to geojson """
        df = pd.DataFrame(d)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
        write_file(gdf, "test.geojson")
        os.remove("test.geojson") # clean up 
        
class TestGetHUC12Boundary():
    
    @pytest.mark.parametrize("huc,valid", [
        ("070802081007", True), 
        ("123456123456", False), 
        ("0708020810", False)
    ])
    def test_get_boundary(self, huc, valid):
        """ 
        test the function passes and fails as expected 
        with valid and invalid HUCs 
        """
        huc12_path = str(Path(HUC12_PATH, f"huc{huc[0:2]}.parquet"))
        output_path = "test.geojson"
        if valid:
            get_huc12_boundary(huc12_path, output_path, huc, "4326")
            assert os.path.exists(output_path)
            os.remove(output_path)
        else:
            with pytest.raises(Exception):
                get_huc12_boundary(huc12_path, output_path, huc, "4326")

    @pytest.mark.parametrize("huc,sr", [
        ("070802081007", "4326"), 
        ("070802081007", "3857"), 
        ("020301030802", "4326"), 
        ("020301030802", "3857")
    ])
    def test_sr_changed(self, huc, sr):
        """ 
        test the function correctly transforms spatial projection
        """
        huc12_path = str(Path(HUC12_PATH, f"huc{huc[0:2]}.parquet"))
        output_path = "test.geojson"
        get_huc12_boundary(huc12_path, output_path, huc, sr)
        huc12 = gpd.read_file(output_path)
        assert sr == str(huc12.crs.to_epsg())
        os.remove(output_path)
       
    @pytest.mark.parametrize("huc,overwrite", [
        ("070802081007", True), 
        ("070802081007", False), 
        ("020301030802", True), 
        ("020301030802", False)
    ])
    def test_overwrite(self, caplog, huc, overwrite):
        """ 
        check the function won't overwrite the 
        extracted huc if it already exists
        """
        huc12_path = str(Path(HUC12_PATH, f"huc{huc[0:2]}.parquet"))
        output_path = "test.geojson"
        sr = "4326"
        get_huc12_boundary(huc12_path, output_path, huc, sr, overwrite=overwrite)
        if overwrite:
            # should be ableo to overwrite 
            get_huc12_boundary(huc12_path, output_path, huc, sr, overwrite=overwrite)
        else:
            # should return logging message
            get_huc12_boundary(huc12_path, output_path, huc, sr, overwrite=overwrite)
            caplog.set_level(logging.INFO)
            assert "file already exists" in caplog.text
         
        # clean up
        os.remove(output_path)
        