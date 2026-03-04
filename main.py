from langchain_ollama import ChatOllama 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 

llm = ChatOllama(
   model= "minimax-m2.5:cloud",
   temperature=0.7 
)

#message = [ SystemMessage(content="?."), HumanMessage(content="What is RAG?")

        
        
prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a helpful assistant."),
     ("human", "{question}")
]) 
 
chain = prompt | llm | StrOutputParser() 
response = chain.invoke({"question": "What is rag?"}) 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{question}") 
])
print(response)