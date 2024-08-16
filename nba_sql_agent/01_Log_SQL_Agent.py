# Databricks notebook source
# MAGIC %pip install --upgrade mlflow==2.14.1 langchain-openai langchain==0.2.6 langchain-community databricks-sql-connector sqlalchemy==2.0 databricks-agents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run `sql_agent_nb` to get access to variables and configs defined in the notebook

# COMMAND ----------

# MAGIC %run ./sql_agent_nb

# COMMAND ----------

# MAGIC %md ### Log SQL Agent Chain to MLflow

# COMMAND ----------

from mlflow.models import infer_signature

input_example = {"messages": [{"role": "user", "content": "How many games have the Boston Celtics won so far this season?"}],"table_desc":table_desc, "column_comments": 
column_comments, "few_shot_examples": few_shot_examples}

prediction = "The Boston Celtics have won 7 games in the past season"

signature = infer_signature(input_example, prediction)

# COMMAND ----------

import mlflow
import os

chain_path = "./sql_agent_nb"

with mlflow.start_run():
    mlflow.log_param("prompt", prompt)
    info = mlflow.langchain.log_model(lc_model=chain_path, 
                                      artifact_path="chain",
                                      input_example=input_example,
                                      signature=signature,
                                      example_no_conversion=True,
                                      extra_pip_requirements=["databricks-agents",
                                                              "langchain-community",
                                                              "databricks-sql-connector",
                                                              "sqlalchemy==2.0",
                                                              "langchain",])

# Load the model and run inference
sql_agent = mlflow.langchain.load_model(model_uri=info.model_uri)

# COMMAND ----------

# MAGIC %md ### Invoke the SQL Agent chain inside our notebook to test that it works

# COMMAND ----------

mlflow.langchain.autolog()

with mlflow.start_run() as run:
  sql_agent.invoke(input_example)

# COMMAND ----------

# MAGIC %md ### Register the SQL Agent Chain to MLflow

# COMMAND ----------

import mlflow
catalog = "main"
schema = "nba_sql_agent"
model_name = "model"

uc_model_name = f"{catalog}.{schema}.{model_name}"
mlflow.set_registry_uri("databricks-uc")
mv = mlflow.register_model(info.model_uri, uc_model_name)

# COMMAND ----------

from mlflow.tracking import MlflowClient

description = "updated SparkSQLToolkit"

client = MlflowClient()

client.update_model_version(
    name=uc_model_name,
    version=mv.version,
    description=description
)

# COMMAND ----------


