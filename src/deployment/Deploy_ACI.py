# Databricks notebook source
# MAGIC %md
# MAGIC # Web Service deployment to Azure Container Instances

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to Azure Resourcs and direct the mlflow API to AzureML

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

# models = [Model(ws, name=model_name) for model_name in ws.models if dbutils.widgets.getArgument('exp_prefix') in model_name]
# get all the models in the ml studio workspace, except not the Riverine models 
models = [Model(ws, name=model_name) for model_name in ws.models if 'Riverine' not in model_name]

#model = Model(ws, 'mlflow-test-aci-model', version=10)
# model_3 = Model(ws, 'floodml-3', version=1)
# model_5 = model_5 = Model(ws, 'floodml-5', version=2)

# COMMAND ----------

# from azureml.core import Environment
# from azureml.core.conda_dependencies import CondaDependencies

# env = Environment.get(workspace=ws, name="AzureML-minimal-ubuntu18.04-py37-cpu-inference")
# # env = Environment.from_dockerfile('floodml', 'docker_deploy/Dockerfile')
# flood_ml_spark_env = env.clone("customize_curated")
# conda_packages = ['pyspark==3.2.0', 'openjdk','pip','pyarrow==3.0.0']
# #Need to include 'pip' above if installing pip packages
# mycondaenv = CondaDependencies.create(conda_packages=conda_packages,python_version='3.7.10') 
# # To access Azure Blob Storage using Python SDK
# mycondaenv.add_pip_package('azure-storage-blob')
# mycondaenv.add_pip_package('delta-spark')
# flood_ml_spark_env.python.conda_dependencies=mycondaenv
# # 'AZURE_STORAGE_CONNECTION_STRING':'DefaultEndpointsProtocol=https;AccountName=insightmlstorage;AccountKey=g8FBMSkNj/BO97w7EeNEz/vjHAP4VdwSDNUggBdHgq8OABoTh5Eh/w11plPZQDuGo/GsChf/aCeY4X76AQ7BDA==;EndpointSuffix=core.windows.net',
# flood_ml_spark_env.environment_variables = {'SCORING_TIMEOUT_MS': '60000',
#                                       'STORAGE_ACCOUNT_NAME': dbutils.widgets.getArgument('storage_account_name'),
#                                       'STORAGE_ACCOUNT_KEY': dbutils.secrets.get(scope = "azureml", key = "blob_key"),
#                                       'CLIENT_ID': dbutils.secrets.get(scope = "azureml", key = "azure_client_id"),
#                                       'CLIENT_SECRET': dbutils.secrets.get(scope = "azureml", key = "azure_client_secret"),
#                                       'TENANT_ID': dbutils.secrets.get(scope = "azureml", key = "azure_tenant_id"),
#                                       'MODEL_PREFIX': dbutils.widgets.getArgument('exp_prefix'),
#                                       'MAX_RUNOFF': str([int(float(model.tags['runoff'])) for model in models][-1])
#                                       }
                                     
# flood_ml_spark_env.inferencing_stack_version='latest'
# flood_ml_spark_env.python.user_managed_dependencies=False

# COMMAND ----------

from azureml.core import Environment

flood_ml_spark_env = Environment.get(ws, 'customize_curated')

# COMMAND ----------

from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

inference_config = InferenceConfig(entry_script="/dbfs/FileStore/floodml_scripts/score_multiple.py",
                                   environment=flood_ml_spark_env) 
aci_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=10, enable_app_insights=False)

service = Model.deploy(workspace=ws,
                       name='flood-predictor',
                       models=models,
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)
service.wait_for_deployment(show_output=True)
print(service.get_logs())

# COMMAND ----------


