# Deployment

Author: Assaad Mrad (assaad.mrad@stantec.com)

Once the model is trained, tested, and [registered](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli#registermodel) (see Model Registration section in the models readme), it is ready for deployment. There are two deployment modes for Stantec Insight: sandbox deployment and production deployment. Sandbox deployment is the normally the first deployment of the model is is done in a [Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace) inside the `insight_sbx_rg` resource group in the `Stantec-DigitalServices-Sandbox` subscription of Azure. Sandbox deployment is used to test the model in a pseudo-production setting and for demos. Production deployment follows after successful sandbox deployment and after weeks or months of testing and demos. Production deployment occurs inside the `connect.io-insight-dev-rg` resource group inside the `Stantec-DigitalServices-Development` subscription of Azure.

By the time of writing, there are three ways to deploy registed models:
- As a web service targeting [Azure Container Instances](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-azure-container-instance) (ACI).
- As a web service targeting [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-azure-kubernetes-service?tabs=python) (AKS). Here's a great [YouTube documentary](https://www.youtube.com/watch?v=BE77h7dmoQU) on kubernetes or k8s. 
- As a [managed online endpoint](https://docs.microsoft.com/en-us/azure/machine-learning/concept-endpoints).

Of these three, Stantec IT has rejected the use of ACI. Here's what Orlando Kelly had to say about it in an email:
>As per the action points here is the update from IT around Azure container instances.
>
>Azure container instances hasn't been onboarded yet, and it actually probably will not. IT found some problems with it, that Microsoft is apparently working on. ACIs cannot pull images from privately networked registries. IT looked for a loophole that the security teams would accept, i.e. using stronger auth (e.g. managed identities), but it requires turning on the "Admin" account on azure container registries, but that requires the container registry to be publicly visible and use poor auth.
>
>There may be some way to make ACIs work and keep the security folks happy, but it's not obvious right now.  And since there are alternatives  (VMs, App Services),  IT are waiting to see if MS addresses ACI's security issues.

AKS is the way to go for production deployments. However, it is overkill for sandbox deployments because it presents myriad challenges detailed in [this middleware.io article](https://middleware.io/blog/kubernetes-challenges-and-solutions/)

Managed online endpoints is chosen as the deployment method for sandbox deployments for its [enhanced MLOps experience compared to AKS](https://docs.microsoft.com/en-us/azure/machine-learning/concept-endpoints#managed-online-endpoints-vs-kubernetes-online-endpoints-preview). It provides automatic infrastructure managed, out-of-box monitoring, logging, and security features. As of the time of writing, Virtual Networks are not supported though Microsoft say they are working on it (managed online endpoints are still in preview mode). This makes managed online endpoints only an interim solution while Microsoft work on supporting VNETs and our infrastructure team works on implementing a solution.

## Sandbox deployment

Sandbox deployment uses Azure's managed online endpoints technology which is still a preview service. To deploy a model using managed online endpoints, [these steps are followed](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints). As of the time of writing, managed online endpoints can only be setup using [Azure Command Line Interface](https://docs.microsoft.com/en-us/cli/azure/) (CLI). There are two entities to managed online endpoint deployments:
- the endpoint,
- and the deployments within that endpoint.

Traffic (as a percentage of total) can be rerouted to specific deployments as is done for [canary deployments](https://towardsdatascience.com/automatic-canary-releases-for-machine-learning-models-38874a756f87) but for the purposed on sandbox deployment this feature will seldom be used. 

There are four main requirements for managed online endpoint deployments:
1. Configuration YAML files
    1. for the endpoint. See [reference YAML schema for endpoing](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-endpoint-managed-online).
    2. for deployments. See [reference YAML schema for deployments](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online).
2. Environment YAML file: this contains additional conda and pip dependencies on top of those provided by the environment image define in the deployments YAML file.
3. [Scoring script](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints#understand-the-scoring-script). See [advanced entry/scoring script authoring](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script) which includes instructions on how to automatically generate a Swagger file for service input and output schema. Any entry/scoring script needs to contain an `init()` method and a `run(Inputs)` method.
4. Models to deploy
    1. could be already registed in the Azure ML worskspace,
    2. or inside a directory in your local path
        - required for deploying multiple models at the same time
        - multi-model deployments [not currently supported](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints#use-more-than-one-model) using managed online endpoints 

I have included the configuration, environment, and scoring used to deploy the `flash-flood-predictor` endpoint and the `multi-model` deployment in this repo.

### Making Spark and `delta-spark` work inside containers

At the bottom of the included scoring script `score_multiple.py` is defined a `get_spark_storage(storage_account_name, application_id, password, tenant_id)` method which return a [SparkSession](https://spark.apache.org/docs/latest/sql-getting-started.html). Flood Predictor accesses stored features inside a [Delta Table](https://docs.delta.io/latest/quick-start.html). To initiate SparkSession access to the Delta Table, we download additional JARs from the Maven repository using this piece of code:

    spark = configure_spark_with_delta_pip(builder) \
    .config("spark.jars.packages", ",".join(["io.delta:delta-core_2.12:1.1.0", "org.apache.hadoop:hadoop-azure:3.3.1", "org.apache.hadoop:hadoop-azure-datalake:3.3.1", "org.wildfly.openssl:wildfly-openssl:2.1.4.Final"])) \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()     

IMPORTANT: The exact version of `delta-core`, `hadoop-azure`, and `hadoop-azure-datalake` need to exactly match the versions of `delta-spark` and `hadoop` downloaded by pip. If deployment fails, you can start a BASH terminal inside the container and navigate to the folder cotaining the JAR files to check which versions are required.

Notice how the SparkSession builder specifies additional spark configurations, read up on those [here](https://docs.delta.io/latest/quick-start.html#set-up-apache-spark-with-delta-lake) and [here](https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/azure/adls-gen2/azure-datalake-gen2-sp-access).