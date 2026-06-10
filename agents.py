from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
)
vector_store = Chroma(
    persist_directory="vectorstore",
    embedding_function = embedding_model
)

retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)

def research_agent(query):
    doc = retriever.invoke(query)
    return doc

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()

def analysis_agent(query,docs):
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
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
    