# Databricks notebook source
# MAGIC %md 
# MAGIC # Install dependencies

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow==2.14.1 langchain==0.2.6 langchain-community databricks-sql-connector sqlalchemy==2.0 databricks-agents typing-extensions langchain_openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F 
import pandas as pd
import mlflow
from mlflow.client import MlflowClient
client = MlflowClient()

# COMMAND ----------

dbutils.widgets.text("mlflow_run_id", "")

# COMMAND ----------

# MAGIC %md ## Build Evaluation Dataset
# MAGIC Evaluation dataset follows the [schema](https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-set.html#eval-set-schema) expected by Mosaic AI Agent Evaluation. We also add an extra field `difficulty` so we can separate evaluation of easier and harder questions

# COMMAND ----------

eval_data = [
    {
        "request": "How many away games did the Boston Celtics win in 2023",
        "difficulty": "easy",
        "expected_response": "Boston Celtics won 20 away games in 2023",
        "expected_retrieved_context": [
            {
                "doc_uri": """SELECT count(season_id) as total_wins,year(game_date) as year 
                                FROM nba_games 
                                WHERE wl_away = "W" and team_name_away = "Boston Celtics" and year(game_date) = 2023 
                                GROUP BY year(game_date)"""
            },
        ],
    },

    {
        "request": "How many times did the Boston Celtics beat the New York Knicks in 2023?",
        "difficulty": "easy",
        "expected_response": "Boston Celtics had zero wins against the New York Knicks in 2023",
        "expected_retrieved_context": [
            {
                "doc_uri": """select sum(case when (team_abbreviation_home = 'BOS' and wl_home = 'W') 
                                or (team_abbreviation_away = 'BOS' and wl_away = 'W') then 1 else 0 end) as boston_celtics_wins
                                from main.jeanne_choo_nba_sql_agent.nba_games
                                where (matchup_home = 'BOS vs. NYK' or matchup_home = 'NYK vs. BOS' or matchup_away = 'BOS vs. NYK' or matchup_away = 'NYK vs. BOS') and year(game_date) = 2023"""
            },
        ],
    },

    {
        "request": "What is the average points spread of games in the first week of June 2023",
        "difficulty": "easy",
        "expected_response": "7",
        "expected_retrieved_context": [
            {
                "doc_uri": """select avg(abs(plus_minus_home)) as points_spread 
                                from nba_games 
                                where `game_date` BETWEEN "2023-06-01" AND "2023-06-07"""
            },
        ],
    }
]

eval_df = spark.createDataFrame(eval_data)

# COMMAND ----------

eval_df.head()

# COMMAND ----------

# MAGIC %md ## Run MLflow Evaluate

# COMMAND ----------

# Log our evaluations to the same run as the one we used for logging our chain. 

mlflow_run_id = getArgument('mlflow_run_id')

with mlflow.start_run(run_id=mlflow_run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{mlflow_run_id}/chain",  
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md ## Analysis

# COMMAND ----------

# MAGIC %md #### We want to see whether or not the agent is able to decide correctly what tool to use

# COMMAND ----------

# helper class 
import mlflow
import json
from mlflow.client import MlflowClient
import pandas as pd
from typing import List
from mlflow.artifacts import download_artifacts


class AgentAnalyzer():
    
    def __init__(self, req_ids, mlflow_run_id) -> None:
        self.req_ids = req_ids
        self.mlflow_run_id = mlflow_run_id

    def get_react_agent_output(self) -> pd.DataFrame:
      """
      Retrieves and aggregates output data from ReActSingleInputOutputParser spans for a list of request IDs.

      This function uses the MLflow client to retrieve trace data for eval request IDs, filters the trace data for spans with the name 
      "ReActSingleInputOutputParser", extracts their outputs, and aggregates these outputs into a single DataFrame. 

      Parameters:
      - req_ids (list): A list of request IDs to query for traces.

      Returns:
      - pd.DataFrame: A DataFrame containing the aggregated outputs from the  
        ReActSingleInputOutputParser spans 
        for each request ID in the input list. Each row represents an output, with a column for the request ID 
        and other columns for the keys present in the output JSON.
      """
      trace_pdfs = pd.DataFrame()
      client = MlflowClient()
      for r in self.req_ids:
        trace = client.get_trace(request_id=r)
        spans = [span.to_dict() for span in trace.data.spans if "ReActSingleInputOutputParser" in span.name]
        outputs = []
        for i in range(len(spans)):
          output_js = json.loads(spans[i]["attributes"]["mlflow.spanOutputs"])
          outputs.append(output_js)
        df = pd.DataFrame(outputs)
        df["request_id"] = r
        trace_pdfs = trace_pdfs.append(df)
      return trace_pdfs

    def count_agent_actions(self, agent_logs: pd.DataFrame) -> pd.DataFrame:
      """
      Counts the number of 'AgentAction' events for each request ID in a given list, using a provided DataFrame of agent logs.

      This function iterates over a list of request IDs, filters the agent logs DataFrame for rows matching each request ID and 
      where the 'type' column equals 'AgentAction'. It then counts the number of such rows (events) for each request ID. 
      The results are aggregated into a new DataFrame with columns for the request ID and the count of 'AgentAction' events.

      Parameters:
      - req_ids (List): A list of request IDs for which to count 'AgentAction' events.
      - agent_logs (pd.DataFrame): A DataFrame containing agent logs, expected to have at least 'request_id' and 'type' columns.

      Returns:
      - pd.DataFrame: A DataFrame with two columns, 'req_id' and 'count', where each row represents a request ID from the input list 
      and its corresponding count of 'AgentAction' events in the agent logs.
      """
      def count_actions(req_id):
        df = agent_logs[(agent_logs["request_id"]==req_id) & (agent_logs["type"]=="AgentAction")]
        count = df.shape[0]
        return count
      
      agent_actions_ls = []

      for req in req_ids:
        count = count_actions(req)
        agent_actions = {"req_id": req, "count": count}
        agent_actions_ls.append(agent_actions)
      agent_actions_df = pd.DataFrame(agent_actions_ls)

      return agent_actions_df
    
    def check_agent_correctness(self, mlflow_artifact_path: str, eval_artifact_save_path: str):
      """
      Extracts correctness evaluations from LLM-as-a-judge results JSON saved to MLflow. Includes ratings and rationales for relevance to query and correctness.
  
      Parameters:
      - mlflow_artifact_path (str): The relative path to the evaluation artifact within the MLflow run artifacts.
      - eval_artifact_save_path (str): The DBFS path where the downloaded artifact should be saved.
  
      Returns:
      - pd.DataFrame: A DataFrame containing selected correctness-related data from the evaluation artifact. Columns include 
        ratings and rationales for relevance to query and correctness.
      """
      def download_eval_json(mlflow_artifact_path, eval_artifact_save_path)-> pd.DataFrame:
          dbutils.fs.mkdirs(eval_artifact_save_path)
          # Download the artifact
          local_path = download_artifacts(
              run_id=self.mlflow_run_id,
              artifact_path=mlflow_artifact_path
          )
          print(f"Artifact downloaded to: {local_path}")
          dbutils.fs.mv(f"file:{local_path}", eval_artifact_save_path)
          eval_artifact_full_path = f"/dbfs{eval_artifact_save_path}/{mlflow_artifact_path}"
          print(f"Artifact moved to {eval_artifact_full_path}")
          return eval_artifact_full_path
      
      pd.set_option('display.max_colwidth', -1)

      eval_artifact_full_path = download_eval_json(mlflow_artifact_path, eval_artifact_save_path)
      evals = pd.read_json(eval_artifact_full_path, orient="split")
      correctness_df = evals[["response/llm_judged/relevance_to_query/rating", "response/llm_judged/relevance_to_query/rationale", "response/llm_judged/correctness/rating", "response/llm_judged/correctness/rationale"]]
      return correctness_df

# COMMAND ----------

# Retrieve a trace by request ID
req_ids = ["XXX", "XXX", "XXX"]    
agent_analyzer = AgentAnalyzer(req_ids, mlflow_run_id)

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflow Traces logs all the Langchain calls that the ｀Langchain SQL Agent Executor｀  makes to answer our question. We can parse these Traces, and extract the outputs of the ｀ReActSingleOutputParser｀ calls to see the sequence of steps our Agent has gone through to answer our question. These steps can be saved into a Pandas Dataframe, from                                                                                                                          which we can see which tools the Agent selected as well as the Agent’s reasoning for selecting that tool. 

# COMMAND ----------

agent_logs = agent_analyzer.get_react_agent_output()

# COMMAND ----------

# MAGIC %md 
# MAGIC Here is a sample of the output of the sequence of steps our Agent has gone through to answer our question. We can see the tools the Agent selected as well as the Agent’s reasoning. 

# COMMAND ----------

agent_logs.head()

# COMMAND ----------

# DBTITLE 1,How many actions does it take an agent to return a response?
agent_analyzer.count_agent_actions(agent_logs)

# COMMAND ----------

# MAGIC %md We want to see whether or not the agent's reasoning makes sense

# COMMAND ----------

eval_artifact_save_path = f"PATH TO MLFLOW ARTIFACTS"
mlflow_artifact_path = "eval_results.json"
agent_analyzer.check_agent_correctness(mlflow_artifact_path, eval_artifact_save_path, )

# COMMAND ----------


