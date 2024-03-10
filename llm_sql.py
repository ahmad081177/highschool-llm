# %% [markdown]
# ### Links

# %% [markdown]
# https://medium.com/data-science-at-microsoft/you-ask-the-questions-gpt-digs-the-insights-2d5760a2b32e

# %% [markdown]
# ### Imports

# %%
import sys, os, json, glob

sys.path.append('.')
sys.path.append('..')

# %%
from io import StringIO
import pickle
import openai
import pandas as pd
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts.chat import  ChatPromptTemplate, PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.llm import LLMChain
from pandas import DataFrame

import pyodbc 
pyodbc.drivers()

# %% [markdown]
# ### Local DB Connection

# %%
#LOCAL SQL
DRIVER='SQL SERVER'
SERVER='ILTLVVMINNOV1\DMSERVER'
DATABASE='MDW Cloud'

connectionString=f"""
driver={{{DRIVER}}};
server={SERVER};
database={DATABASE};
Trusted_Connection=yes;
"""

conn = pyodbc.connect(connectionString)

# %% [markdown]
# ### Remote DB Connection

# %%
SERVER = 'ILTLVVMINNOV1\DMSERVER'
DATABASE = 'MDW Cloud'
USERNAME = 'sa'
PASSWORD = 'P@ssword2019'
DRIVER = 'SQL SERVER'
#DRIVER = 'ODBC Driver 18 for SQL Serve'
#Trusted_Connection=yes;

connectionString=f"""
driver={{{DRIVER}}};
server={SERVER};
database={DATABASE};
UID={USERNAME};
PWD={PASSWORD};
"""

# %% [markdown]
# ### Open AI Const & Connection

# %%

class __GPT4:
    # TODO move to .env
    API_BASE = "https://openai-aiattack-001257-eastus2-01.openai.azure.com/"
    # TODO move to .env
    API_KEY = "7d0xxxx07c"
    API_VERSION = "2023-07-01-preview"
    API_TYPE = "azure"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    DEPLOYMENT_MODEL = "gpt-4"
    MODEL_NAME = "gpt-4"

def init_openai():
    openai.api_version = GPT.API_VERSION
    os.environ["OPENAI_API_VERSION"] = GPT.API_VERSION

    openai.api_key = GPT.API_KEY
    os.environ["OPENAI_API_KEY"] = GPT.API_KEY

    openai.api_base = GPT.API_BASE
    os.environ["OPENAI_API_BASE"] = GPT.API_BASE

    openai.api_type = GPT.API_TYPE
    os.environ["OPENAI_API_TYPE"] = GPT.API_TYPE

GPT = __GPT4

# %%
def create_AzureChat(temp: float = 0) -> ChatOpenAI:
    llm = AzureChatOpenAI(
        temperature=temp,
        deployment_name=GPT.DEPLOYMENT_MODEL,
        model_name=GPT.MODEL_NAME,
        openai_api_base=GPT.API_BASE,
        openai_api_key=GPT.API_KEY,
        openai_api_version=GPT.API_VERSION,
        verbose=False,
    )
    return llm

def create_qa(llm: ChatOpenAI, db:FAISS):
    qa = RetrievalQA.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=False,
    )
    return qa

# %%
init_openai()
embeddings = GPT4AllEmbeddings()
llm0 = create_AzureChat()

# %% [markdown]
# ### DB Functions

# %%
def open_db_connection(connStr: str = connectionString) ->pyodbc.Connection:
    return pyodbc.connect(connStr)

def close_connection(conn:pyodbc.Connection):
    conn.cursor().close()
    conn.close()

# %%
def execute_sql(conn: pyodbc.Connection, sql_query:str):
    # print('===')
    # print(sql_query)
    # print('===')
    cursor = conn.cursor()
    cursor.execute(sql_query)
    records = cursor.fetchall()
    return records, cursor.description

# %%
def get_table_columns(conn: pyodbc.Connection, db:str, table:str) -> list:
    if not db.startswith('['):
        db = '[' + db + ']'

    query = f"SELECT TOP (0) * FROM {db}.{table}"
    
    record,desc = execute_sql(conn, query)
    cols = [d[0] for d in desc]
    return cols
    

# %%
def get_display_name(s: str) -> str:
    return "".join(map(lambda x: x if x.islower() else " " + x, s)).strip()


# %% [markdown]
# ### Opcenter Constants

# %%
SITE_ID=2
YEAR_OF_INTEREST=2022

NUM_OF_RECORDS = 100
DBNAME=f'[{DATABASE}]'


# %%
from enum import Enum
class OpcenterTables(Enum):
    OperationRequest='[bm20].[OperationRequest]'
    OperationResponse = '[bm20].[OperationResponse]'
    OperationScheduling = '[bm20].[OperationScheduling]'
    OperationExecution = '[bm20].[OperationExecution]'
    OperationExecutionStatus = '[bm20].[OperationExecutionStatus]'
    OperationResponseEquipmentSpecification = '[bm20].[OperationResponseEquipmentSpecification]'
    Equipment = '[bm20].[Equipment]'
    
    #[OperationExecutionPropertyStaticValue]
    #[PropertyType]

# %% [markdown]
# ### Opcenter SQL Queries

# %%

def select_query_str(tablename:str, dbname:str = DBNAME, num_of_record:int = NUM_OF_RECORDS) ->str:
  query=f"""
SELECT TOP ({num_of_record}) *  FROM {dbname}.{tablename}
Where ([OperationRequestSiteId]={SITE_ID} AND YEAR([StartDateTime])>={YEAR_OF_INTEREST})
"""
  return query

# %% [markdown]
# ### Main

# %% [markdown]
# Prepare docs embedding

# %%
from llm_utils.embeddings import load_db, save_docs_to_db
from llm_utils.files2text import file2docs

OPCENTER_MAIN_FOLDER='../../data/opcenter'
DOCS_FOLDER = OPCENTER_MAIN_FOLDER+'/documents/'
DOCS_EMBED_DB = OPCENTER_MAIN_FOLDER +'/docs_db'


# %%
## Create Embedding
# docs = file2docs(DOCS_FOLDER)
# print(f'num of docs: {len(docs)}')
# docs_db = save_docs_to_db(docs, embeddings, DOCS_EMBED_DB)

## Load Embedding
docs_db = load_db(DOCS_EMBED_DB, embeddings)

# %% [markdown]
# DB: Get Tables/Columns

# %%
conn = pyodbc.connect(connectionString)

# %%
## Example Query
sql_str = select_query_str(OpcenterTables.OperationRequest.value, DBNAME, 10)
operation_request_records, _ = execute_sql(conn, sql_str)
operation_request_records

# %% [markdown]
# Table Desc using LLM

# %%
## Eeither Load or Generate
fname=OPCENTER_MAIN_FOLDER+'/table_defs.p'


def create_table_and_cols(conn):
    # Use LLM to create Table descriptions - This takes arond 15 seconds (depends on number of tables.)
    qa = create_qa(create_AzureChat(), docs_db)

    #tables and cols
    tables_and_cols = {}
    for t in OpcenterTables:
        cols = get_table_columns(conn, DBNAME , t.value)
        nicename = get_display_name(t.name)
        
        query_t = f'What is the definition of {nicename} Entity. Summarize it in few sentences, no more than 40 words.'
        ret = qa({'query' : query_t })
        desc = ret['result']
        
        tables_and_cols[t.value] = {'cols': cols,'nicename': nicename, 'desc': desc}
    return tables_and_cols
        

## Create Table and Cols
# conn = open_db_connection()
# tables_and_cols = create_table_and_cols(conn)
# close_connection(conn)
# pickle.dump(tables_and_cols, open(fname,'wb'))

## Load 'Table and Cols'
tables_and_cols = pickle.load(open(fname,'rb'))



# %%
for t in tables_and_cols:
    print(f'Table: {tables_and_cols[t]["nicename"]} -- {tables_and_cols[t]["desc"]}' )
    print(tables_and_cols[t]['cols'])

# %% [markdown]
# GPT

# %%
def ask_me(llm, messages:list):
    ret = llm(messages=messages)
    
    print("LLM Answer:\n========")
    print(ret)

    conn = open_db_connection()
    try:
        r,_ = execute_sql(conn, ret.content)
        print("\nData from DB:\n=======")
        for r1 in r:
            print(r1)
    finally:
        close_connection(conn)

# %%
system_context = '''
You are T-SQL expert, can generate high quality T-SQL script,
double check T-SQL gramma, double check T-SQL syntex.
Don't explain the code, just generate the code block itself
'''


table_desc="-- Your SQL should be using the following SQL tables only.\n-- Table names and descriptions:\n"
for t in tables_and_cols:
    nicename = tables_and_cols[t]['nicename']
    desc = tables_and_cols[t]['desc']
    table_desc +=f"--    {nicename} : {desc}\n"

table_def="-- Table Definitions:\n"
for t in tables_and_cols:
    name = str(t)
    #name = tables_and_cols[t]['nicename']
    cols = tables_and_cols[t]['cols']
    table_def +=f"--    {name}(: {cols})\n"

bginfo = f'''

-- Important background details about our SQL and Data:
--    OperationSchedule table represents the planning phase for work order. 
--    OperationExecution table represents the actual processing and executing of a work order.
--    Each work order is split into small phases, this is done using OperationRequest and OperationResponse using OperationScheduleId and OperationExecutionId columns.
--    Table T1 reference to table T2 is done by adding column T2Id in table T1.
'''

dataset_conext = f'''
-- T-SQL Script, Microsoft SQL Server.

{table_desc}

{table_def}

{bginfo}
'''


user_query = 'What workorders are the problematic ones (biggest delay), display worst 10 in addition to the execution status column'
user_input = f"""
-- T-SQL script to {user_query}, 
-- Use minimal table required. 
-- Build the script step by step with comments,
"""

from langchain.schema.messages import HumanMessage, SystemMessage
um = HumanMessage(content= user_input)
sm = SystemMessage(content=system_context)
sm2 = SystemMessage(content=dataset_conext)
messages=[
    sm,sm2,um
]


# %%
ask_me(llm0, messages)

# %% [markdown]
# Query#2

# %%
#user_query = 'find me work orders that are close enough to OperationExecutionKey="154102", i.e. have some similarity but not neccessarly same exact values'
user_query = 'Find the top-10 work orders that have been finished too early'
user_input = f"""
-- T-SQL script to {user_query},
-- Filter your SQL query by SiteId=2
-- Filter your SQL query only from 2022 and forward
-- Use minimal table required. 
-- Build the script step by step with comments,
"""

from langchain.schema.messages import HumanMessage, SystemMessage
um = HumanMessage(content= user_input)
sm = SystemMessage(content=system_context)
sm2 = SystemMessage(content=dataset_conext)
messages=[
    sm,sm2,um
]

# %%
ask_me(llm0, messages)

# %%
user_query = 'What open workorder are in risk for delay'
user_input = f"""
-- T-SQL script to {user_query},
-- Filter your SQL query by SiteId=2
-- Filter your SQL query only from 2022 and forward
-- Use minimal table required. 
-- Build the script step by step with comments,
"""

from langchain.schema.messages import HumanMessage, SystemMessage
um = HumanMessage(content= user_input)
sm = SystemMessage(content=system_context)
sm2 = SystemMessage(content=dataset_conext)
messages=[
    sm,sm2,um
]

# %%
ask_me(llm0, messages)

# %%
user_query = 'What open workorder are in risk for delay'
user_input = f"""
-- T-SQL script to {user_query},
-- Filter your SQL query by SiteId={SITE_ID}
-- Filter your SQL query only from {YEAR_OF_INTEREST} and forward
-- The criteria of workorders at risk is the num of units produced so far vs units to produce.
-- Use minimal table required. 
-- Build the script step by step with comments.
-- Double check your SQL is 100% correct.
"""

from langchain.schema.messages import HumanMessage, SystemMessage
um = HumanMessage(content= user_input)
sm = SystemMessage(content=system_context)
sm2 = SystemMessage(content=dataset_conext)
messages=[
    sm,sm2,um
]

# %%
ask_me(llm0, messages)

# %% [markdown]
# User Journey:
# Wost performance...Actual vs Planned
#   Rate: #of products per hours
#   Setup: Time to prepare the machine
#   Productive time: time the machine was supposed to work, 3 shifts  VS it worked 2 shifts  (Availability ROE)
# 
# 
# 
# 
#   All WO produces BOP
# 
# 
# 
# Work order of the same kind that are comparable - same properties...(Final Material, Input Material, Asset/Equipement? )
#  ** XXX Rate XXXX
#  Name prop of OperatiohExecution: Final Material (Exact label)
#  Name prop of OperationScheduling

# %% [markdown]
# Find the worst work order (more delays)
# 
# 
# 
# Find the work orders that finished too early
# 
# 
# 
# Work order of the same kind that are comparable 
# ** same properties...(Final Material, Input Material, Asset/Equipement? )
#  ** XXX Rate XXXX
#  Name prop of OperatiohExecution: Final Material (Exact label)
#  Name prop of OperationScheduling
# 
# 
# What open workorder are in risk for delay...
# "Open" has actual start-time but no end time (OperationExecution)
# 
#    Units produced so far vs units to produce


