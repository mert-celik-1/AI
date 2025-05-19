import os
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.chat_models import init_chat_model
from langchain.schema.runnable.base import RunnableMap
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import  Chroma
from langchain_openai import OpenAIEmbeddings
from openai import embeddings
from langchain.memory import ConversationBufferMemory


load_dotenv(override=True)

"""

model = init_chat_model("gpt-4o-mini",model_provider="openai")

messages = [
    SystemMessage("Verilen metni İngilizceden Türkçeye çevir."),
    HumanMessage("Hello world")
]

response = model.invoke(messages)
print(response.content)
"""


"""
Prompt Template

model = init_chat_model("gpt-4o-mini",model_provider="openai")

system_template = "Verilen metni İngilizceden {dil} diline çevir."
user_template = "{metin}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system",system_template),("user",user_template)]
)

prompt = prompt_template.invoke({"dil":"Almanca","metin":"hello world"})

response = model.invoke(prompt)
print(response.content)
"""

"""
Retrieve & Semantic Search

pdf_loader = PyPDFLoader("attention.pdf")

pdf_docs = pdf_loader.load()

web_loader = WebBaseLoader("https://cbarkinozer.medium.com/transformat%C3%B6rler-t%C3%BCm-i%CC%87htiyac%C4%B1n%C4%B1z-olan-dikkat-ee4ff66723b1")
web_docs = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n", "\n", " ", ""])

docs = text_splitter.split_documents(pdf_docs + web_docs)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings)

query = "Explain Multi-Head Attention "
results = vectorstore.similarity_search(query,k=3)
print(results)

"""


"""
Basic Memory

memory = ConversationBufferMemory(return_messages=True)

memory.chat_memory.add_message(HumanMessage("Merhaba!"))
memory.chat_memory.add_message(AIMessage("Size nasıl yardımcı olabilirim?"))
memory.chat_memory.add_message(HumanMessage("İstanbul hakkında bilgi verir misin?"))

print(memory.load_memory_variables({}))
"""

