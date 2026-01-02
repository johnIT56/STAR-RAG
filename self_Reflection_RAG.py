#!/usr/bin/env python
# coding: utf-8

# In[5]:


from dotenv import load_dotenv
load_dotenv()


# In[6]:


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import chromadb


# In[7]:



import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load the existing collection
client = chromadb.PersistentClient(path="./knowledge-base-collection")
embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma(
    client=client,
    collection_name='knowledge-base-collection',
    embedding_function=embedding_function
)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 3})


# In[8]:


from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str
    
graph_builder = StateGraph(AgentState)


# In[9]:


def retrieve(state: AgentState) -> AgentState:
    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}


# In[10]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini')


# In[11]:


from langchain import hub 

generate_prompt = hub.pull('rlm/rag-prompt')

generate_llm = ChatOpenAI(model='gpt-4o', max_completion_tokens=100) #why max token 100?

def generate(state: AgentState) -> AgentState:
    context = state['context']
    query = state['query']
    
    rag_chain = generate_prompt | generate_llm
    
    response = rag_chain.invoke({'question':query,'context':context})
    
    return {'answer': response.content}


# In[12]:


from typing import Literal

doc_relevance_prompt = hub.pull('langchain-ai/rag-document-relevance')

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    query = state['query']
    context = state['context']
    
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question':query, 'documents': context})
    
    if response['Score'] == 1:
        return 'relevant'
    return 'irrelevant'


# In[13]:


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

rewrite_prompt = PromptTemplate.from_template("""
You are a helpful query rewriting assistant for a Retrieval-Augmented Generation (RAG) system.

Your goal:
- Make the user’s question clearer and more retrieval-friendly *only if truly necessary*.
- If the question is already clear, specific, and meaningful, do NOT rewrite it.
- Avoid trivial rewordings or restating the same idea in different words.
- Do not enter a rewrite loop — stop rewriting if the query already makes sense.

Output format:
Return ONLY the final query text (no explanations or reasoning).

User question: {{query}}
Rewritten query:
""")

def rewrite(state: AgentState) -> AgentState:
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query': query})
    return {'query': response}


# In[14]:


from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from kdu(kyungdong university) global campus, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})

    return response
     


# In[15]:


# LangChain 허브에서 유용성 프롬프트를 가져옵니다
helpfulness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState) -> str:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다.

    Args:
        state (AgentState): 사용자의 질문과 생성된 답변을 포함한 에이전트의 현재 state.

    Returns:
        str: 답변이 유용하다고 판단되면 'helpful', 그렇지 않으면 'unhelpful'을 반환합니다.
    """
    # state에서 질문과 답변을 추출합니다
    query = state['query']
    answer = state['answer']

    # 답변의 유용성을 평가하기 위한 체인을 생성합니다
    helpfulness_chain = helpfulness_prompt | llm
    
    # 질문과 답변으로 체인을 호출합니다
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})

    # 점수가 1이면 'helpful'을 반환하고, 그렇지 않으면 'unhelpful'을 반환합니다
    if response['Score'] == 1:
        return 'helpful'
    
    return 'unhelpful'

def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수입니다. 
    graph에서 conditional_edge를 연속으로 사용하지 않고 node를 추가해
    가독성을 높이기 위해 사용합니다

    Args:
        state (AgentState): 에이전트의 현재 state.

    Returns:
        AgentState: 변경되지 않은 state를 반환합니다.
    """
    # 이 함수는 현재 아무 작업도 수행하지 않으며 state를 그대로 반환합니다
    return state


# In[16]:


graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)


# In[17]:


from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END ##INSTEAD OF ENDING WE CAN ADD LLM THAT ASK USER TO CLARIFY THE QN
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)
graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite','retrieve')


# In[18]:


graph = graph_builder.compile()


# In[19]:


from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))


# In[ ]:





# In[ ]:





# ## Notes
# 
# future improvements:
# ->for inhouse use if not rellevant instead of endtask ask user to clarify the question
