import os 
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print('Initializing components')
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
  """Answer the question based only on the following context:
  {context}

  Question: {question}

  Provide a detailed answer:"""
)

def format_docs(docs):
  """Format retrieved documents into a single string."""
  return "\n\n".join(doc.page_content for doc in docs)


#==============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
#==============================================================================
def create_retrieval_chain_with_lcel():
  """
  Create a retrieval chain using LCEL (LangChain Expression Language)
  Returna a chain that can be invoked with {"question": "..."}

  Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
  """

  retrieval_chain = (
    RunnablePassthrough.assign(
      context=itemgetter("question") | retriever | format_docs 
    ) |
    prompt_template
    | llm 
    | StrOutputParser()
  )
  return retrieval_chain





if __name__ == '__main__':
  print('Retrieving....')

  #query
  query = "What is Pinecone in maching learning?"

  chain_with_lcel = create_retrieval_chain_with_lcel()
  result_with_lcel = chain_with_lcel.invoke({'question': query})
  print('\nAnswer:')
  print(result_with_lcel)