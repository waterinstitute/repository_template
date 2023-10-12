# Databricks notebook source
# MAGIC %md 
# MAGIC # Register logged experiment run to Azure Machine Learning Model Repository
# MAGIC 
# MAGIC References: 
# MAGIC 
# MAGIC - Deploy machine learning models to Azure https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python#registermodel
# MAGIC - Experiment class https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment(class)?view=azure-ml-py
# MAGIC - Run class https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Set up widgets to connect to Azure Machine Learning

# COMMAND ----------

# dbutils.widgets.text(name="aml_subscription_id", defaultValue="87ea213f-502b-4fca-8120-34b9b9e22789", label="AML subscription ID")
# dbutils.widgets.text(name="aml_resource_group", defaultValue="Stantec-DigitalServices-Sandbox", label="AML resource group")
# dbutils.widgets.text(name="aml_name", defaultValue="Insight-Machine-Learning", label="AML name")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup authentication and connection to Azure Workspace

# COMMAND ----------

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
# MAGIC ### Retrieve experiment in which run of interest exists

# COMMAND ----------

# MAGIC %md Get all experiments that start with the prefix specified in the deployment pipeline

# COMMAND ----------

from azureml.core import Model
from azureml.exceptions import WebserviceException

exp_prefix = dbutils.widgets.getArgument('exp_prefix')
# exp_prefix = 'Flood_Predictor'
experiments = ws.experiments

for exp in ws.experiments:
    if exp_prefix in exp:
        runs = experiments[exp].get_runs()
        
        optimizing_metric_best = 0
        best_run = None
        for run in runs:
            if run.get_status() == 'Completed':
#             print(run.get_metrics()["F1 score"])
                optimizing_metric = run.get_metrics()[dbutils.widgets.getArgument('optimizing_metric')]
#             optimizing_metric = run.get_metrics()['F1 score']
                if optimizing_metric > optimizing_metric_best:
                    best_run = run
                    optimizing_metric_best = optimizing_metric
                
        model_name = exp
        register = False # Boolean to decide wether to register the model
        try:
            model = Model(ws, name = model_name)
        except WebserviceException:
            print(model_name, 'experiment does not exist. Creating new model with this name.')
            register = True
            
        if not register and model.run_id != best_run.id:
            print('Latest model version,', model.version,', has different run ID. Registering improved model.')
            register = True
        
        if register:
            model_path = 'model'
            tags = best_run.get_tags()

            model = best_run.register_model(model_name, model_path=model_path, tags=tags)
        else:
            print('Latest model version for', model_name,', version', model.version,', already has best run.')
        
#         print(list(runs)[0])

# COMMAND ----------


