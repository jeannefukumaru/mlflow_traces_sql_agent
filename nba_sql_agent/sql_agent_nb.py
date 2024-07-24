# Databricks notebook source
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
import mlflow
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
import os

# COMMAND ----------

# add prompts into a list for future reference. When saving the chain, we choose the first (latest) prompt in the list by default
prompts = [
  PromptTemplate(
    template="""You are an expert in the NBA Basketball league. Your task is to answer user questions about games. You have access to the following tables:
    - table containing game statistics

    Think step-by-step what SQL query you need in order to answer the users question. Include the SQL query you finally used in your final output. 

    User's question: {question}
    
    ### SQL QUERY USED""",
    input_variables=["question"],),
  
  PromptTemplate(
    template="You are an expert in the NBA Basketball league. Your task is to answer user questions about games  User's question: {question}",
    input_variables=["question"],)]

# COMMAND ----------

# These helper functions parse the `messages` array.
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]

input_example = {"messages": [{"role": "user", "content": "How many times did the Boston Celtics win an away game this year?"}]}

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct", max_tokens=500)

db = SQLDatabase.from_databricks(catalog="main", schema="nba_sql_agent", host="", warehouse_id="")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_executor_kwargs={"handle_parsing_errors":True})

prompt = prompts[0]

chain = (
    {
        "question": itemgetter("messages")
        | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | prompt
    | agent
    | RunnableLambda(lambda x: x["output"])
    | StrOutputParser()
)

mlflow.models.set_model(chain)

# COMMAND ----------


