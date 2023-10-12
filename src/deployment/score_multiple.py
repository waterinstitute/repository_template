'''
References for entry script authoring:
- Advanced entry script authoring https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script

'''

import numpy as np
import pandas as pd
import os
from flask import jsonify

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from pyspark.ml.functions import vector_to_array
    
from delta import *

from azureml.core.model import Model
# from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel 
from pyspark.sql.functions import asc, posexplode, arrays_zip, col
from pyspark.sql.types import *

def init():
    global models
    global spark
    global path
    
    storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
    storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
    application_id = os.getenv('CLIENT_ID')
    password = os.getenv('CLIENT_SECRET')
    tenant_id = os.getenv('TENANT_ID')

    # Set up Spark Session with delta lake
    spark = get_spark_storage(storage_account_name, application_id, password, tenant_id)

    # Load pypark pipeline models
    
    models=[]
    for runoff in range(int(os.getenv('MAX_RUNOFF'))):
        model_name = os.getenv('MODEL_PREFIX')+'_'+str(runoff+1)+'_0'
        print('Loading model, '+model_name+', for runoff of '+str(runoff+1))
        model_path = Model.get_model_path(model_name=model_name)
        models.append(PipelineModel.load(os.path.join(model_path,'sparkml')))

    
    # load path to storage
    path = "abfss://{}@{}.dfs.core.windows.net/{}".format(str('flood-ml-data'), str(storage_account_name), str("grass_data"))


# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators parses and validates the incoming payload against
# the example input. This will also generate a Swagger
# API document for the web service.

runoff_sample_input = StandardPythonParameterType(2.3)
huc_sample_input = StandardPythonParameterType('071200040505')
row_sample_input = StandardPythonParameterType(5)
column_sample_input = StandardPythonParameterType(8)

# This is a nested input sample, any item wrapped by `ParameterType` will be described by schema
sample_input = StandardPythonParameterType({'runoff': runoff_sample_input,
                                           'huc': huc_sample_input, 
                                           'row': row_sample_input, 
                                           'column': column_sample_input})

pred_sample_output = StandardPythonParameterType(1.0)
prob_sample_output = StandardPythonParameterType(0.6072714)


sample_output = StandardPythonParameterType({"prediction": pred_sample_output, "probability": prob_sample_output})
outputs = StandardPythonParameterType({"Results": sample_output})

@input_schema('Inputs', sample_input)
@output_schema(outputs)
def run(Inputs):
    '''
    for this iteration, data should contain:
    1- runoff
    2- huc
    3- row
    4- column
    '''
    
    print(Inputs)
    
    # Unpack dictionary
    runoff = float(Inputs['runoff'])
    huc = Inputs['huc']
    row = int(Inputs['row'])
    column = int(Inputs['column'])

    df_raw_load = spark.read.format("delta").load(path)
    
    if huc != '' and row >=0 and column >=0:


        df_raw = df_raw_load.where("huc={}".format(huc) ).where("idx_0={}".format(row) ).where("idx_1={}".format(column) ).coalesce(1)

    
        layers = ['twi', 'geofloodindex_local', 'finalelevationstream_local', 'distancestream_local', 'finalcontribarea_local', 'geofloodindex_nonlocal',
                  'finalelevationstream_nonlocal', 'distancestream_nonlocal', 'finalcontribarea_nonlocal']

    
        df_build = df_raw.select(posexplode(arrays_zip(*layers)).alias("local_pos","x")).select("x.*")
        
        model = models[round(runoff)-1]
        # model_name = os.getenv('MODEL_PREFIX')+'_'+str(int(runoff))+'_0'
        # print('Loading model, '+model_name+', for runoff of '+str(runoff))
        # model_path = Model.get_model_path(model_name=model_name)
        # model = PipelineModel.load(os.path.join(model_path,'sparkml'))
    
        df_predictions = model.transform(df_build)
    
        pandas_df = df_predictions.withColumn('probability_1',vector_to_array('probability')).select('prediction',col('probability_1')[1]).toPandas()
    
        pred_list = pandas_df['prediction'].tolist()
        prob_list = pandas_df['probability_1[1]'].tolist()
        
        dims = df_raw.select('dims').collect()[0][0]
        
        return jsonify({'dims': dims, 'prediction': pred_list, 'probability': prob_list})
        
    elif huc == '':

        pandas_df = df_raw_load.where("idx_0=0" ).where("idx_1=0").select('huc').toPandas()

        return jsonify({'huc_choices': pandas_df['huc'].tolist()})

    elif row < 0:
        df = df_raw_load.where("huc={}".format(huc)).where("idx_1=0")
        lat_min = df.select(df.lat[0].alias('lat_min')).agg({'lat_min': 'min'}).collect()[0][0]
        lat_max = df.select(df.lat[1].alias('lat_max')).agg({'lat_max': 'max'}).collect()[0][0]
        lat_dim = df.select(df.dims[0].alias('lat_dims')).agg({'lat_dims': 'sum'}).collect()[0][0]
#         print(df.printSchema())
        
        pandas_df = df.select(['idx_0', 'crs']).coalesce(1).sort(asc("idx_0")).toPandas()
        
        return jsonify({'row_choices': pandas_df['idx_0'].tolist(), 'lat_range': [lat_min, lat_max], 'lat_dim': lat_dim , 'crs': pandas_df['crs'].iloc[0]})
    
    elif column < 0:
        df = df_raw_load.where("huc={}".format(huc)).where("idx_0=0")
        long_min = df.select(df.long[0].alias('long_min')).agg({'long_min': 'min'}).collect()[0][0]
        long_max = df.select(df.long[1].alias('long_max')).agg({'long_max': 'max'}).collect()[0][0]
        long_dim = df.select(df.dims[1].alias('long_dims')).agg({'long_dims': 'sum'}).collect()[0][0]
        
        pandas_df = df.select(['idx_1', 'crs']).coalesce(1).sort(asc("idx_1")).toPandas()
    
        return jsonify({'column_choices': pandas_df['idx_1'].tolist(), 'long_range': [long_min, long_max], 'long_dim': long_dim, 'crs': pandas_df['crs'].iloc[0]})
    
    else:
        
        return jsonify({})

def get_spark_storage(storage_account_name, application_id, password, tenant_id):
    """
    Gets a spark session with a connection to an Azure storage account 
    and the ability to write to Delta. 
    If the session has already been created, it will be returned.

    Resources: # https://docs.delta.io/latest/quick-start.html https://docs.delta.io/latest/delta-storage.html
    """
    builder = SparkSession.builder.appName("Flood_Predict") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    
    spark = configure_spark_with_delta_pip(builder) \
    .config("spark.jars.packages", ",".join(["io.delta:delta-core_2.12:1.1.0", "org.apache.hadoop:hadoop-azure:3.3.1", "org.apache.hadoop:hadoop-azure-datalake:3.3.1", "org.wildfly.openssl:wildfly-openssl:2.1.4.Final"])) \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()     
    
    spark.conf.set("fs.azure.account.auth.type.{}.dfs.core.windows.net".format(storage_account_name), "OAuth")
    spark.conf.set("fs.azure.account.oauth.provider.type.{}.dfs.core.windows.net".format(storage_account_name),  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
    spark.conf.set("fs.azure.account.oauth2.client.id.{}.dfs.core.windows.net".format(storage_account_name), application_id)
    spark.conf.set("fs.azure.account.oauth2.client.secret.{}.dfs.core.windows.net".format(storage_account_name), password)
    spark.conf.set("fs.azure.account.oauth2.client.endpoint.{}.dfs.core.windows.net".format(storage_account_name), "https://login.microsoftonline.com/{}/oauth2/token".format(tenant_id))

    return spark

