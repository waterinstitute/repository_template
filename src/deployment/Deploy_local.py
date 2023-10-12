# Databricks notebook source
# MAGIC %md
# MAGIC # Local Model Web Service
# MAGIC 
# MAGIC References:
# MAGIC - https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/deploy-to-local
# MAGIC - https://docs.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-deployment-local

# COMMAND ----------

# MAGIC %md
# MAGIC ### Declare variables needed to connect to Azure Resources

# COMMAND ----------

import os

#os.environ["CONNECTION_STRING"]='DefaultEndpointsProtocol=https;AccountName=dtistorm2;AccountKey=FSG75JiiO8w0KzEfFfcTXwHGgN5P4pdewY5A2VEwrqUaxf056d1QytvMG17HXShl9GhSRtle5yPpLuDluLrXQw==;EndpointSuffix=core.windows.net'
#os.environ["CONTAINER_NAME"]='storm'

#Service Principal
os.environ["azure_tenant_id"]='413c6f2c-219a-4692-97d3-f2b4d80281e7'
os.environ["azure_client_id"]='7e94e67a-a7ce-4ed2-aaaa-4cc7c94da047'
os.environ["azure_client_secret"]='V~cafa7SZt_6IwYqT2HeiGX56Y6Q3E_e-y'
#Azure ML ID
os.environ["aml_subscription_id"]='eb6f96f0-ff9e-450e-b2a6-5519de91feb9'
os.environ["aml_resource_group"]='BC2258-DigitalSolutions-Workbench'
os.environ["aml_name"]='Insight-Machine-Learning'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to Azure Resourcs and direct the mlflow API to AzureML

# COMMAND ----------

#https://databricks.com/blog/2020/12/22/natively-query-your-delta-lake-with-scala-java-and-python.html
# import pandas, deltalake, mlflow, os
import pandas, os
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
# from mlflow.tracking.client import MlflowClient


sp = ServicePrincipalAuthentication(
    tenant_id = os.environ["azure_tenant_id"],
    service_principal_id = os.environ["azure_client_id"],
    service_principal_password = os.environ["azure_client_secret"])
ws = Workspace.get(
  name= os.environ["aml_name"],
  location = 'eastus',
  subscription_id=os.environ["aml_subscription_id"],
  resource_group=os.environ["aml_resource_group"], 
  auth=sp
  )

# set up connection to Azure ML for logging 
# uri = ws.get_mlflow_tracking_uri()
# mlflow.set_tracking_uri(uri)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pull in registered model from Azure ML

# COMMAND ----------

from azureml.core.model import Model

#model = Model(ws, 'mlflow-test-aci-model', version=10)
model = Model(ws, 'mlflow-test-aci-model', version=16)

# COMMAND ----------

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment.get(workspace=ws, name="AzureML-minimal-ubuntu18.04-py37-cpu-inference")
# env = Environment.from_dockerfile('floodml', 'docker_deploy/Dockerfile')
curated_clone = env.clone("customize_curated")
conda_packages = ['pyspark==3.1.1', 'openjdk','pip']
#Need to include 'pip' above if installing pip packages
mycondaenv = CondaDependencies.create(conda_packages=conda_packages,python_version='3.7.10') 
# To access Azure Blob Storage using Python SDK
mycondaenv.add_pip_package('azure-storage-blob')
# The below speeds up conda environment solving significantly
mycondaenv.remove_channel('conda-forge')
mycondaenv.add_pip_package('delta-spark')
# mycondaenv.add_pip_package('azure-storage-file-datalake>=12.0.0')
curated_clone.python.conda_dependencies=mycondaenv
curated_clone.environment_variables = {'SCORING_TIMEOUT_MS': '7000000', 
                                       'AZURE_STORAGE_CONNECTION_STRING':'DefaultEndpointsProtocol=https;AccountName=insightmlstorage;AccountKey=g8FBMSkNj/BO97w7EeNEz/vjHAP4VdwSDNUggBdHgq8OABoTh5Eh/w11plPZQDuGo/GsChf/aCeY4X76AQ7BDA==;EndpointSuffix=core.windows.net',
                                      'STORAGE_ACCOUNT_NAME':'insightmlstorage',
                                      'STORAGE_ACCOUNT_KEY': 'g8FBMSkNj/BO97w7EeNEz/vjHAP4VdwSDNUggBdHgq8OABoTh5Eh/w11plPZQDuGo/GsChf/aCeY4X76AQ7BDA=='}
curated_clone.inferencing_stack_version='20210623.40134510'
curated_clone.python.user_managed_dependencies=False
# curated_clone

# COMMAND ----------

from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script="score.py",
                                   environment=curated_clone)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Necessary step to work on Windows
# MAGIC 
# MAGIC Windows restricts the length of the file names automatically. Some files in the model artifact have a long name which creates an error. To fix, I started 1)Run, 2)typed Regedit, and 3)changed the 'LongPathsEnabled' key to 1 from 0 to not throw an error. I followed step 3 from here https://www.techinpost.com/file-path-too-long/

# COMMAND ----------

from azureml.core.webservice import LocalWebservice

# This is optional, if not provided Docker will choose a random unused port.
deployment_config = LocalWebservice.deploy_configuration(port=6789)

local_service = Model.deploy(ws, "test", [model], inference_config, deployment_config)

local_service.wait_for_deployment()

# COMMAND ----------

print(local_service.get_logs())

# COMMAND ----------

import json

sample_input = json.dumps({
    'huc': '071200040505',
    'row': 5,
    'column': 8
})


# COMMAND ----------

# MAGIC %%time
# MAGIC response = local_service.run(sample_input)

# COMMAND ----------

print(response[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Updating Deployment

# COMMAND ----------

local_service.reload()
print("--------------------------------------------------------------")
# After calling reload(), run() will return the updated message.
results = local_service.run(sample_input)

# COMMAND ----------

results[:10]

# COMMAND ----------

# MAGIC %%time
# MAGIC results = local_service.run(sample_input)

# COMMAND ----------

print(results[:10])

# COMMAND ----------

print(local_service.get_logs())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Steps to access blob storage from Docker container: 
# MAGIC 
# MAGIC ##### Method 1: Using blobfuse
# MAGIC 
# MAGIC - Install `software-properties-common`: https://phoenixnap.com/kb/add-apt-repository-command-not-found-ubuntu
# MAGIC - Configured linux software repository for microsoft products: https://docs.microsoft.com/en-us/windows-server/administration/Linux-Package-Repository-for-Microsoft-Software
# MAGIC - Moved the cfg file which I have on the host into the docker container inside the /root directory
# MAGIC - installed curl: `apt install curl`
# MAGIC - Installed libcurl3: `apt-get install libcurl3-gnutls`
# MAGIC - Requires running the container with additional privileges: `--cap-add=SYS_ADMIN --device /dev/fuse` https://stackoverflow.com/questions/53951624/azure-container-instances-with-blobfuse-or-azure-storage-blobs
# MAGIC 
# MAGIC This issue with Method 1 is that there doesn't exist a way to use fuse from inside the Docker container without running it with increased privileges. Doing so with Azure is currently not supported and IT might not like that approach anyway. Check out this GitHub issue for details: https://github.com/docker/for-linux/issues/321
# MAGIC 
# MAGIC ##### Method 2: Using the Azure Blob Storage Python SDK and sending API request for required data
# MAGIC 
# MAGIC - Install pip dependency: `delta-spark`
# MAGIC - Install pip dependency: `azure-storage-blob`
# MAGIC - Add `AZURE_STORAGE_CONNECTION_STRING` as an environment variable
# MAGIC - Accessing blob storage from python SDK: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python
# MAGIC - Set up Blob Service Client: `blob_service_client = BlobServiceClient.from_connection_string(connect_str)`
# MAGIC - Get container client: `flood_ml_data_client = blob_service_client.get_container_client('flood-ml-data')`
# MAGIC - Get blob generator based on chosen huc, row, and column: `blobs = flood_ml_data_client.list_blobs(name_starts_with=f'grass_data/huc={huc}/idx_0={row}/idx_1={column}/part')`
# MAGIC - Check out the `BlobClient` class API for information on how to download chosen blobs. The score.py script contains working code.
# MAGIC 
# MAGIC ##### Method 3: Using DeltaTable and the where method
# MAGIC 
# MAGIC CRITICAL: Setting up Spark to work correctly with Delta
# MAGIC - Configure spark with delta pip: `spark = configure_spark_with_delta_pip(builder)` as explained in https://docs.delta.io/latest/quick-start.html#python
# MAGIC - Add jars for delta-core, hadoop-azure, hadoop-azure-datalake, and wildfly-openssl: `spark = spark = configure_spark_with_delta_pip(builder).config("spark.jars.packages", ",".join(["io.delta:delta-core_2.12:1.0.0", "org.apache.hadoop:hadoop-azure:3.2.0", "org.apache.hadoop:hadoop-azure-datalake:3.2.0", "org.wildfly.openssl:wildfly-openssl:2.1.4.Final"])).getOrCreate()`. VERY IMPORTANT: versions of hadoop-azure and hadoop-azure-datalake need to match the hadoop jar versions already present in the spark lib
# MAGIC - Set spark azure configs:
# MAGIC 
# MAGIC ```
# MAGIC spark.conf.set("fs.azure.account.auth.type.{}.dfs.core.windows.net".format(storage_account_name), "OAuth")
# MAGIC spark.conf.set("fs.azure.account.oauth.provider.type.{}.dfs.core.windows.net".format(storage_account_name),  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
# MAGIC spark.conf.set("fs.azure.account.oauth2.client.id.{}.dfs.core.windows.net".format(storage_account_name), application_id)
# MAGIC spark.conf.set("fs.azure.account.oauth2.client.secret.{}.dfs.core.windows.net".format(storage_account_name), password)
# MAGIC spark.conf.set("fs.azure.account.oauth2.client.endpoint.{}.dfs.core.windows.net".format(storage_account_name), "https://login.microsoftonline.com/{}/oauth2/token".format(tenant_id))
# MAGIC ```
# MAGIC 
# MAGIC based on https://docs.delta.io/latest/delta-storage.html#microsoft-azure-storage for Azure Data Lake Storage Gen2
# MAGIC 
# MAGIC - Define base path to storage in init file using abfss

# COMMAND ----------


