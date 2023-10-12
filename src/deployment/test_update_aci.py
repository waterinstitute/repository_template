# Databricks notebook source
# MAGIC %md
# MAGIC # Test AzureML deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to AzureML workspace

# COMMAND ----------

import pandas, os
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace


sp = ServicePrincipalAuthentication(
  tenant_id = dbutils.secrets.get(scope = "azureml", key = "azure_tenant_id"),
  service_principal_id = dbutils.secrets.get(scope = "azureml", key = "azure_client_id"),
  service_principal_password = dbutils.secrets.get(scope = "azureml", key = "azure_client_secret"))
ws = Workspace.get(
  name= dbutils.secrets.get(scope = "azureml", key = "workspace.name"),
  location = 'eastus',
  subscription_id=dbutils.secrets.get(scope = "azureml", key = "subscription.id"), 
  resource_group=dbutils.secrets.get(scope = "azureml", key = "resource.group"), 
  auth=sp
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pull in registered model from Azure ML

# COMMAND ----------

from azureml.core.model import Model

#model = Model(ws, 'mlflow-test-aci-model', version=10)
model_3 = Model(ws, 'mlflow-test-aci-model', version=18)
model_5 = model_5 = Model(ws, 'floodml-5', version=1)

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
curated_clone.environment_variables = {'SCORING_TIMEOUT_MS': '60000', 'AZURE_STORAGE_CONNECTION_STRING':'DefaultEndpointsProtocol=https;AccountName=insightmlstorage;AccountKey=g8FBMSkNj/BO97w7EeNEz/vjHAP4VdwSDNUggBdHgq8OABoTh5Eh/w11plPZQDuGo/GsChf/aCeY4X76AQ7BDA==;EndpointSuffix=core.windows.net',
                                      'STORAGE_ACCOUNT_NAME':'insightmlstorage',
                                      'STORAGE_ACCOUNT_KEY': 'g8FBMSkNj/BO97w7EeNEz/vjHAP4VdwSDNUggBdHgq8OABoTh5Eh/w11plPZQDuGo/GsChf/aCeY4X76AQ7BDA==',
                                      'CLIENT_ID': '7e94e67a-a7ce-4ed2-aaaa-4cc7c94da047',
                                      'CLIENT_SECRET': 'V~cafa7SZt_6IwYqT2HeiGX56Y6Q3E_e-y',
                                      'TENANT_ID': '413c6f2c-219a-4692-97d3-f2b4d80281e7'}
curated_clone.inferencing_stack_version='20210623.40134510'
curated_clone.python.user_managed_dependencies=False
# curated_clone

# COMMAND ----------

from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

inference_config = InferenceConfig(entry_script="/dbfs/FileStore/floodml_scripts/score_multiple.py",
                                   environment=curated_clone) 
aci_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=10, enable_app_insights=True)

service = Model.deploy(workspace=ws,
                       name='flood-predictor',
                       models=[model_3, model_5],
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)
service.wait_for_deployment(show_output=True)

# COMMAND ----------

# print(service.get_logs())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Get Webservice in case we are just updating

# COMMAND ----------

from azureml.core import Webservice

service = Webservice(ws, 'flood-predictor-aks')
print(service)

# COMMAND ----------

# MAGIC %md ### Enable App Insights

# COMMAND ----------

# service.update(enable_app_insights=True)

# COMMAND ----------

import json

data ={ 'Inputs': {
    'runoff': 4.2,
    'huc': '',
    'row': 5,
    'column': 8
}}

body = str.encode(json.dumps(data))

# COMMAND ----------

import urllib.request

url = 'http://20.75.132.91:80/api/v1/service/flood-predictor-aks/score'

api_key = 'hDuCoJfP9x5jdWOAZznn4zrkTS4JglAm' # Replace this with the API key for the web service

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))

# COMMAND ----------

response = service.run(sample_input)

# COMMAND ----------

response

# COMMAND ----------

# MAGIC %md
# MAGIC ### Updating Deployment

# COMMAND ----------

from azureml.core.environment import Environment

flood_ml_spark_env = Environment.get(ws, 'AzureML-minimal-ubuntu18.04-py37-cpu-inference')

# COMMAND ----------

from azureml.core.model import InferenceConfig
inference_config = InferenceConfig(entry_script="/dbfs/FileStore/floodml_scripts/score_test_connectivity.py",
                                   environment=flood_ml_spark_env) 

# COMMAND ----------

service.update(inference_config=inference_config)
service.wait_for_deployment(show_output=True)

# COMMAND ----------

# MAGIC %%time
# MAGIC results = service.run(sample_input)

# COMMAND ----------

print(results)

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


