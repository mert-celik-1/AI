import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema.runnable.base import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import  Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

model = init_chat_model("gpt-4o-mini",model_provider="openai")

pdf_paths = ["attention.pdf", "avelandia.pdf"]

pdf_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    pdf_docs.extend(loader.load())

web_loader = WebBaseLoader("https://cbarkinozer.medium.com/transformat%C3%B6rler-t%C3%BCm-i%CC%87htiyac%C4%B1n%C4%B1z-olan-dikkat-ee4ff66723b1")
web_docs = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n", "\n", " ", ""])
chunks = text_splitter.split_documents(documents=pdf_docs + web_docs)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks,embeddings,persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Görevin kullanıcı sorularına cevap vermek. Aşağıdaki içeriği referans olarak kullan: {context}"),
    ("user", "{question} {language} dilinde yaz" )
])

def get_query(input_dict):
    return input_dict["question"]

chain = RunnableMap({
    "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
    "question" : lambda x : x["question"],
    "language": RunnableLambda (lambda x: x["language"])
}) | prompt_template | model | StrOutputParser()

response = chain.invoke({"language": "Türkçe","question":"Avelandiyanın kuruluş tarihi nedir"})
print(response)
