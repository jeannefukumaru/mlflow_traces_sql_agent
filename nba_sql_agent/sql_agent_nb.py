# Databricks notebook source
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit, SparkSQLToolkit
from langchain.sql_database import SQLDatabase
from langchain import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
from langchain_openai import ChatOpenAI
import mlflow
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
import os

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Metadata to include with the model
# MAGIC Since we are only using one table, it's more efficient to include the table and column description into the prompt rather than to rely on tools to get table metadata. Cutting down on the number of tool calls can help us reduce query latency

# COMMAND ----------

import json
table_desc = json.dumps({"description": "The 'nba_games' table contains data about NBA games, including details about the teams, game statistics. It includes information such as the game date, matchup, and win-loss record. This data can be used for various purposes, including game analysis. It can also be used to identify trends and patterns in player performance and team performance over time."})

column_comments = json.dumps({'season_id': "Unique identifier for the NBA season.",
 'team_id_home': "Identifier for the home team in the game.",
 'team_abbreviation_home': "Abbreviated representation of the home team's name.",
 'team_name_home': "Full name of the home team in the game.",
 'game_id': "Unique identifier for the game.",
 'game_date': "Date when the game was played.",
 'matchup_home': "Opposing team's abbreviation for the home team.",
 'wl_home': "Win-loss record of the home team.",
 'min': "Minutes played by the player in the game.",
 'fgm_home': "Number of field goals made by the player for the home team.",
 'fga_home': "Number of field goal attempts by the player for the home team.",
 'fg_pct_home': "Field goal percentage for the player's performance in the game.",
 'fg3m_home': "Number of three-point field goals made by the player for the home team.",
 'fg3a_home': "Number of three-point field goal attempts by the player for the home team.",
 'fg3_pct_home': "Three-point field goal percentage for the player's performance in the game.",
 'ftm_home': "Number of free throws made by the player for the home team.",
 'fta_home': "Number of free throw attempts by the player for the home team.",
 'ft_pct_home': "Free throw percentage for the player's performance in the game.",
 'oreb_home': "Number of offensive rebounds by the player for the home team.",
 'dreb_home': "Number of defensive rebounds by the player for the home team."})

few_shot_examples = json.dumps([{'Question': 'How many away games did the Boston Celtics win in 2023', 
                                 'Answer': """SELECT count(season_id) as total_wins,year(game_date) as year 
                                FROM nba_games 
                                WHERE wl_away = "W" and team_name_away = "Boston Celtics" and year(game_date) = 2023 
                                GROUP BY year(game_date)"""}])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Format prompt using `ChatPromptTemplate` 
# MAGIC Now that our prompt is more complex, we can use the Langchain `ChatPromptTemplate` so we can organize the system prompt, user prompt and chat history separately

# COMMAND ----------

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

system_message_template = """"You are an expert in the NBA Basketball league. Your task is to answer user questions about games, teams and players. 

    Think step-by-step what SQL query you need in order to answer the users question. Include the SQL query you finally used in your final output. 

    This is a description of the table: {table_desc}

    This is an explanation of the column comments: {column_comments}

    Here are some examples of the user question and the corresponding SQL query: 

    {few_shot_examples}"""

system_message = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["table_desc", "column_comments", "few_shot_examples"], template=system_message_template))

chat_prompt_str = ChatPromptTemplate([system_message,
 MessagesPlaceholder(variable_name='chat_history', optional=True),
 HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))])

# COMMAND ----------

# add prompts into a list for future reference. When saving the chain, we choose the first a (latest) prompt in the list by default
prompts = [
  chat_prompt_str,
  
  PromptTemplate(
    template="""You are an expert in the NBA Basketball league stats. Your task is to answer user questions about games. You have access to the following tables:
    - table containing game statistics

    Think step-by-step what SQL query you need in order to answer the users question. Include the SQL query you finally used in your final output. 

    User's question: {question}
    
    ### SQL QUERY USED""",
    input_variables=["question"],),
  
  PromptTemplate(
    template="You are an expert in the NBA Basketball league. Your task is to answer user questions about games  User's question: {question}",
    input_variables=["question"],)]

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Helper functions to help us parse our user questions for the LLM Agent

# COMMAND ----------

# These helper functions parse the `messages` array.
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]

# Parse player names if shortforms or nicknames are used
def parse_player_names(query):
    known_nicknames = ["iceman", "air jordan"]
    nicknames_in_query = [n for n in known_nicknames if n in query]
    player_nickname_map = {"iceman": "George Gervin", "air jordan": "Michael Jordan"}
    
    for n in nicknames_in_query:
        try:
            real_name = player_nickname_map[n]
            query = query.replace(n, real_name)
        except KeyError:
            pass

    return query

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Customise tool prompts
# MAGIC Some tools used by our agent have prompts associated with them, for example, the `QuerySparkSQLTool` contains a prompt to check the validity of a sql query before it is executed. If we see persistent model errors because of certain syntax errors, we can update this prompt to be more relevant
# MAGIC

# COMMAND ----------

# from langchain_community.tools.spark_sql.prompt import QUERY_CHECKER
# import pprint
# pprint.pprint(QUERY_CHECKER)

# COMMAND ----------

NEW_QUERY_CHECKER = """
{query}
Double check the Spark SQL query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Make sure your string matching formatting is correct

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query."""

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Update the `SparkSQLToolkit`
# MAGIC The toolkit that we pass to our Agent is really a class that encapsulates a list of tools. Instead of having a tool to list the tables and table information from our database, which takes time, we:  
# MAGIC
# MAGIC 1. include this information directly in our prompt and leave out the original `ListSparkSQLTool` and `InfoSparkSQLTool` 
# MAGIC
# MAGIC 2. update the `QuerySparkSQLChecker` tool to use our updated prompt

# COMMAND ----------

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.spark_sql.tool import (
    QueryCheckerTool,
    QuerySparkSQLTool,
)
from langchain_community.utilities.spark_sql import SparkSQL
from typing import Any, Dict, Optional, List
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

class BaseSparkSQLTool(BaseModel):
    """Base tool for interacting with Spark SQL."""

    db: SparkSQL = Field(exclude=True)

    class Config(BaseTool.Config):
        pass


class NewQueryCheckerTool(BaseSparkSQLTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = NEW_QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "query_checker_sql_db"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    """

    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "llm_chain" not in values:
            from langchain.chains.llm import LLMChain

            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),  # type: ignore[arg-type]
                prompt=PromptTemplate(
                    template=NEW_QUERY_CHECKER, input_variables=["query"]
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool need to use ['query'] as input_variables "
                "for the embedded prompt"
            )

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            query=query, callbacks=run_manager.get_child() if run_manager else None
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query, callbacks=run_manager.get_child() if run_manager else None
        )

class SparkSQLToolkit(BaseToolkit):
  """Toolkit for interacting with Spark SQL.
  Parameters:
      db: SparkSQL. The Spark SQL database.
      llm: BaseLanguageModel. The language model.
  """
  db: SparkSQL = Field(exclude=True)
  llm: BaseLanguageModel = Field(exclude=True)
  
  class Config:
      arbitrary_types_allowed = True

  def get_tools(self) -> List[BaseTool]:
    """Get the tools in the toolkit."""
    return [
        QuerySparkSQLTool(db=self.db),
        NewQueryCheckerTool(db=self.db, llm=self.llm),
    ]

# COMMAND ----------


from langchain_community.agent_toolkits.spark_sql.base import create_spark_sql_agent


input_example = {"messages": [{"role": "user", "content": "How many times did the Boston Celtics win an away game this year?"}], "table_desc":table_desc, "column_comments": column_comments, "few_shot_examples": few_shot_examples}

spark_sql = SparkSQL(catalog="main", schema="nba_sql_agent")
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
tools = SparkSQLToolkit(db=spark_sql, llm=llm)
agent = create_spark_sql_agent(llm=llm, toolkit=tools, verbose=True, agent_executor_kwargs={"handle_parsing_errors":True})
prompt = prompts[0]

chain = (
    {
        "input": itemgetter("messages")
        | RunnableLambda(extract_user_query_string) | RunnableLambda(parse_player_names),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "table_desc": itemgetter("table_desc"),
        "column_comments": itemgetter("column_comments"),
        "few_shot_examples": itemgetter("few_shot_examples"),
    }
    | prompt
    | agent
    | RunnableLambda(lambda x: x["output"])
    | StrOutputParser()
)

mlflow.models.set_model(chain)

# COMMAND ----------


