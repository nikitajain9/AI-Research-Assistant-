from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vector_store = Chroma(
    persist_directory="vectorstore",
    embedding_function = embedding_model
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()


@tool 
def vector_retriever(query:str):
    """This tool is to perform the semantic serach"""
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    docs = retriever.invoke(query)
    results = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    return results

@tool
def bm25_retriever(query: str) -> str:
    """This tool is to perform keyword search"""
    # 1. Fetch contents safely from Chroma
    chroma_data = vector_store.get()
    raw_texts = chroma_data.get('documents', [])
    
    # 2. Safety check: If Chroma is empty, return a message instead of crashing
    if not raw_texts:
        return "Keyword search status: The underlying database is empty. No documents found."
        
    # 3. Safely map text chunks into Document schemas
    bm_docs = [Document(page_content=doc) for doc in raw_texts]
    
    bm25 = BM25Retriever.from_documents(bm_docs)
    bm25.k = 3
    retrieved_docs = bm25.invoke(query)
    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

def research_agent(query):
    messages = []
    llm_with_tools = llm.bind_tools([vector_retriever,bm25_retriever])
    prompt = ChatPromptTemplate.from_messages([
        ('system','Act as an research agent and retrieve relevant document for the given query'),
        ('user','{query}')
    ])

    final_prompt = prompt.invoke({'query':query})

    messages.extend(final_prompt.to_messages())

    response = llm_with_tools.invoke(final_prompt)

    messages.append(response)

    print(response.tool_calls)

    if not response.tool_calls:
        return response.content

    tools = {
    "vector_retriever": vector_retriever,
    "bm25_retriever": bm25_retriever
    }
    context = []

    for tool_call in response.tool_calls:
        tool_name = tool_call['name']
        tool_response = tools[tool_name].invoke(tool_call['args'])
        context.append(tool_response)
    return "\n\n".join(context)

def analysis_agent(query,context):
    prompt = ChatPromptTemplate.from_messages([
        ('system','Act like a helpful assistant and answer the query from the given documents only.'),
        ('user','Question \n{query}\nContext \n{context}')
    ])
    chain = prompt | llm | parser
    result = chain.invoke({'query':query,'context':context})
    return result

def reviewer_agent(query, answer):
    prompt = ChatPromptTemplate.from_template('context:Question: {query}\nAnswer: {answer}\nReview the answer.\nIf information is missing,improve the answer.')
    chain = prompt | llm | parser
    response = chain.invoke({'query':query,'answer':answer})
    return response
    