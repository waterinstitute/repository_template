# Databricks notebook source
# MAGIC %md
# MAGIC # Feature selection
# MAGIC 
# MAGIC For both riverine and flash flooding
# MAGIC Use Obiwlan for high speed comuptation (5mins run)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Import libraries

# COMMAND ----------

from delta.tables import DeltaTable  
import pandas as pd
import itertools

# import azureml.mlflow
import sys, os
import pathlib as pl
import numpy as np
from scipy.optimize import minimize
from scipy import  stats
from matplotlib import pyplot as plt
import seaborn as sns

# from pyspark.sql.functions import *

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler



# COMMAND ----------

# MAGIC %md ## Initialize custom methods

# COMMAND ----------

# MAGIC %md Import pyspark sql functions needed for custom methods

# COMMAND ----------

from pyspark.sql.functions import sum, broadcast, posexplode, arrays_zip, floor, when, col

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md ## Connect to Azure Workspace

# COMMAND ----------

secret_scope ="azureml"
aml_region="eastus"

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
  
  
sp = ServicePrincipalAuthentication(
    tenant_id = dbutils.secrets.get(scope =secret_scope, key = "azure_tenant_id"),
    service_principal_id = dbutils.secrets.get(scope = secret_scope, key = "azure_client_id"),
    service_principal_password = dbutils.secrets.get(scope = secret_scope, key = "azure_client_secret"))
ws = Workspace.get(
    name= dbutils.secrets.get(scope = secret_scope, key = "workspace.name"), #dbutils.widgets.get("aml_name") 
    location = aml_region,
    subscription_id=dbutils.secrets.get(scope =secret_scope, key = "subscription.id"),  # dbutils.widgets.get("aml_subscription_id")
    resource_group= dbutils.secrets.get(scope =secret_scope, key = "resource.group"), #dbutils.widgets.get("aml_resource_group") 
    auth=sp
  )

# COMMAND ----------

# MAGIC %md ## Mount storgate account

# COMMAND ----------

storage_account = "insightmlstoragesbx"
storage_container= "flood-ml-data"
storage_secret_key ="blob_key"

mnt = mount_storage_account(
  storage_container, 
  storage_account, 
  secret_scope, 
  storage_secret_key)

# COMMAND ----------

# MAGIC %md ## Load data (riverine or flash)

# COMMAND ----------

#flood_type = 'flash'
#random_seed = 1
#dataset_name = "flash_data"

# COMMAND ----------

try:
    flood_type = dbutils.widgets.getArgument("flood_type")
    if flood_type == 'flash':
        dataset_name = "flash_data"
    elif flood_type == 'riverine':
        dataset_name = "riverine_data"
except Py4JJavaError as e:
    print(str(e))
    flood_type = "riverine"
    dataset_name = "riverine_data"

print('flood_type:', flood_type)
print('dataset_name:', dataset_name)

# COMMAND ----------

# retrieve an existing Dataset in the workspace by name
from azureml.core import Dataset

latest_riverine_dataset = Dataset.get_by_name(ws, dataset_name, version='latest') 
dataset_tags = latest_riverine_dataset.tags

print(dataset_tags)

# COMMAND ----------

df = latest_riverine_dataset.to_spark_dataframe()
df = df.na.drop()

if flood_type == 'flash':
    runoff_chosen=4
    df = df \
    .withColumn("label", when(df["threshold_runoff"] < runoff_chosen, 1).otherwise(0)) 
    non_features = ['huc', 'rows', 'cols','threshold_runoff','label']

elif flood_type == 'riverine':
    non_features = ['huc', 'rows', 'cols', 'label']
    
features = [c for c in df.columns if c not in non_features]
label= ['label']
layers = features + label
print('layers:', layers)

# COMMAND ----------

df = df.select(layers)

try:
    random_seed = int(dbutils.widgets.getArgument("random_seed"))
except Py4JJavaError as e:
    print(str(e))
    random_seed = 1
    
train, test = df.randomSplit([0.95, 0.05],seed=random_seed)

# COMMAND ----------

#df = df.select(layers)
#train, test = df.randomSplit([0.95, 0.05],seed=random_seed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regularization Methods for Feature Selection

# COMMAND ----------

from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#Make sure the normalization is for each column
assembler = VectorAssembler(inputCols=features, outputCol="assembled")
scaler = StandardScaler(withMean=True, withStd=True, inputCol="assembled", outputCol="features")
pipeline = Pipeline(stages=[assembler, scaler])
scalerModel = pipeline.fit(train)
scaledData = scalerModel.transform(train)
df_scaled = scaledData.select("features","label")
df_scaled = df_scaled.withColumn("label",df_scaled.label.cast('int'))

# COMMAND ----------

def reverse_bisect_right(a, x, lo=0, hi=None):
    #the function returns number of elements in a which are >= than x.
    # e,g.: a=[8,6,4,2,1], b=5
    #reverse_bisect_right(a,b)=2
    #Source: https://stackoverflow.com/questions/2247394/bisect-is-it-possible-to-work-with-descending-sorted-lists
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    return lo

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
print(features)
regParam = pd.Series([0, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.5, 0.75])
l_num = regParam.count()
f_num=len(features)
Coef= np.zeros((l_num,f_num))

# enumerate through regParam with index and i
for ind, i in enumerate(regParam):
    lr_elastic = LogisticRegression(regParam=i, elasticNetParam=0.8,maxIter=100)
    elastic_model = lr_elastic.fit(df_scaled)
    Coef[ind,:] = elastic_model.coefficients.toArray()
  
plt.figure(figsize=(9, 4))
plt.plot(np.log(regParam), Coef)
plt.xlabel('Log_regParam'); plt.ylabel('Coefficients')
plt.legend(features,loc="lower right")
plt.grid()

if flood_type == 'flash':
    plt.savefig(r"/dbfs/FileStore/feature_selection/first_attempt_flash.jpg")
elif flood_type == 'riverine':
    plt.savefig(r"/dbfs/FileStore/feature_selection/first_attempt_riverine.jpg")

# COMMAND ----------

try:
    top_num = int(dbutils.widgets.getArgument("top_num"))
except Py4JJavaError as e:
    print(str(e))
    top_num = 7
    
best_index=None
trigger=False
num_nonzero_fea=np.zeros(l_num)

for i in range(l_num):
    num_nonzero_fea[i]=np.count_nonzero(Coef[i,:])
    if num_nonzero_fea[i]==top_num:
        best_index=i
        trigger=True
print(num_nonzero_fea)

# COMMAND ----------

if best_index == None:
    left_ind=reverse_bisect_right(num_nonzero_fea,top_num)-1
    right_ind=left_ind+1
    reg_left=regParam[left_ind]
    reg_right=regParam[right_ind]
    
    regParam_narrow = pd.Series(np.linspace(reg_left,reg_right,num=l_num))
    l_num_narrow = regParam_narrow.count()
    Coef_narrow = np.zeros((l_num_narrow,f_num))

    # enumerate through regParam with index and i
    for ind, i in enumerate(regParam_narrow):
          lr_elastic = LogisticRegression(regParam=i, elasticNetParam=0.8, maxIter=100)
          elastic_model = lr_elastic.fit(df_scaled)
          Coef_narrow[ind,:] = elastic_model.coefficients.toArray()

    plt.figure(figsize=(9, 4))
    plt.plot(regParam_narrow, Coef_narrow)
    plt.xlabel('regParam_narrow'); plt.ylabel('Coefficients')
    plt.legend(features,loc="lower right")
    plt.grid()
    if flood_type == 'flash':
        plt.savefig(r"/dbfs/FileStore/feature_selection/second_attempt_flash.jpg")
    elif flood_type == 'riverine':
        plt.savefig(r"/dbfs/FileStore/feature_selection/second_attempt_riverine.jpg")   
    
    num_nonzero_fea_narrow=np.zeros(l_num_narrow)

    for i in range(l_num_narrow):
        num_nonzero_fea_narrow[i]=np.count_nonzero(Coef_narrow[i,:])
        if num_nonzero_fea_narrow[i]==top_num:
            best_index=i
    print(num_nonzero_fea_narrow)

# COMMAND ----------

import warnings

if best_index == None:
    nearest_approx=min(num_nonzero_fea_narrow, key=lambda x:abs(x-top_num))
    list_num_nonzero_fea_narrow=list(num_nonzero_fea_narrow)
    best_index=list_num_nonzero_fea_narrow.index(nearest_approx)
    Coef_fea=Coef_narrow
    warnings.warn("Select the number of features that is cloest to the desired number of features")
    
else:
    if trigger == True:
        Coef_fea=Coef
    else:
        Coef_fea=Coef_narrow

selected_coef=Coef_fea[best_index,:]
nonzero_feature_index=np.nonzero(selected_coef)[0]
feature_selected=[features[i] for i in nonzero_feature_index]
num_select=len(feature_selected)

print(feature_selected)
print("We selected {} features when {} features were planned to be selected".format(num_select,top_num))

# COMMAND ----------

if flood_type == 'flash':
    with open('/dbfs/FileStore/feature_selection/flash_features.txt', 'w') as f:
        f.write(str(feature_selected + [random_seed]))
elif flood_type == 'riverine':
    with open('/dbfs/FileStore/feature_selection/riverine_features.txt', 'w') as f:
        f.write(str(feature_selected + [random_seed]))
