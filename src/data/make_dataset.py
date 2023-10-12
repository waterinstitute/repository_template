#!/usr/bin/python 
import argparse 
import click
import inspect
import logging
import sys
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.append(parentdir)
if os.getenv("AZ_BATCH_TASK_WORKING_DIR") is not None:
    sys.path.append(os.getenv("AZ_BATCH_TASK_WORKING_DIR"))

from config import create_config
import process_inputs 
import grass_stats
import utils 

def main(huc12, overwrite):
    """ 
    Runs data processing scripts to turn raw data into data to be
    used for machine learning.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("getting config for huc12: " + huc12)
    config = create_config(huc12)
    
    logger.info("making dataset for config: " + str(config))
    
    logger.info("----------processing inputs----------")
    updated_config = process_inputs.main(huc12, config, overwrite)
    
    logger.info("----------running grass calculations----------")
    grass_stats.main(huc12, updated_config)
    
if __name__ == "__main__":
    log_fmt = "[%(asctime)s] [%(levelname)s] [%(module)s] : %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    parser = argparse.ArgumentParser()
    loc = parser.add_mutually_exclusive_group(required=True)
    loc.add_argument("--point", "-pt", help="a point in lat/long coordinates", nargs=2, type=float)
    loc.add_argument("--huc12", "-huc", help="a huc12 code", type=str)
    parser.add_argument(
        "--overwrite", 
        help="should input files be redownloaded/calculated and overwritten if they exist", 
        dest="overwrite", action="store_true"
    )
    
    args = parser.parse_args()
    
    if args.point:
        huc12 = utils.get_huc12(tuple(args.point))
    else:
        huc12 = args.huc12
        
    main(huc12, args.overwrite)
