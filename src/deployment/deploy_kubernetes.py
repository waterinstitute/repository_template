# Databricks notebook source
# MAGIC %md
# MAGIC # Web Service deployment to Azure Kubernetes Service

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

# MAGIC %md
# MAGIC ### Attach AKS cluster to AML workspace

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget
# Set the resource group that contains the AKS cluster and the cluster name
resource_group = 'insight_sbx_rg'
cluster_name = 'insightsbxaks'

# Attach the cluster to your workgroup. If the cluster has less than 12 virtual CPUs, use the following instead:
# attach_config = AksCompute.attach_configuration(resource_group = resource_group,
#                                         cluster_name = cluster_name,
#                                         cluster_purpose = AksCompute.ClusterPurpose.DEV_TEST)
attach_config = AksCompute.attach_configuration(resource_group = resource_group,
                                         cluster_name = cluster_name)
aks_target = ComputeTarget.attach(ws, 'insightsbxaks', attach_config)

# Wait for the attach process to complete
aks_target.wait_for_completion(show_output = True)

# COMMAND ----------

from azureml.core import Environment

flood_ml_spark_env = Environment.get(ws, 'customize_curated')

# COMMAND ----------

from azureml.core.model import InferenceConfig

# Uncomment line below to deploy models
# inference_config = InferenceConfig(entry_script="/dbfs/FileStore/floodml_scripts/score_multiple.py",
#                                    environment=flood_ml_spark_env) 

# Uncomment line below to test AKS connectivity
inference_config = InferenceConfig(entry_script="/dbfs/FileStore/floodml_scripts/score_test_connectivity.py",
                                   environment=flood_ml_spark_env) 

# COMMAND ----------

from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import Model
from azureml.core.compute import AksCompute


aks_target = AksCompute(ws,"insightsbxaks")
# If deploying to a cluster configured for dev/test, ensure that it was created with enough
# cores and memory to handle this deployment configuration. Note that memory is also used by
# things such as dependencies and AML components.
deployment_config = AksWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 10, scoring_timeout_ms=300000, auth_enabled=False)
service = Model.deploy(ws, "flood-predictor-aks2", [models[0]], inference_config, deployment_config, aks_target)
service.wait_for_deployment(show_output = True)
print(service.state)
print(service.get_logs())

# COMMAND ----------


