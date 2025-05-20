import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema.runnable.base import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig, RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
import gradio as gr

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

model = init_chat_model("gpt-4o-mini",model_provider="openai",streaming=True)

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
if os.path.exists("./chroma_db"):
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
retriever = vectorstore.as_retriever()

def format_docs(docs):
    content = "\n\n".join(doc.page_content for doc in docs)
    return clean_text(content)

message_histories = {}

def get_session_history(session_id):
    if session_id not in message_histories:
        message_histories[session_id] = ChatMessageHistory()
    return message_histories[session_id]


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Görevin kullanıcı sorularına cevap vermek. Aşağıdaki içeriği referans olarak kullan: {context}"),
    ("system", "Önceki konuşma geçmişi: {chat_history}"),
    ("user", "{question} {language} dilinde yaz")
])


def clean_text(text):
    return text.encode('utf-8', errors='ignore').decode('utf-8')


def get_query(input_dict):
    return input_dict["question"]

chain = RunnableMap({
    "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
    "question" : lambda x : x["question"],
    "language": RunnableLambda (lambda x: x["language"]),
    "chat_history": lambda x: x.get("chat_history", "")
}) | prompt_template | model | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)


def respond(message, chat_history, language):
    session_id = "gradio_session"

    chat_history.append({"role": "user", "content": message})

    chat_history.append({"role": "assistant", "content": ""})

    yield chat_history, ""

    response_parts = []

    for chunk in chain_with_history.stream(
            {"language": language, "question": clean_text(message)},
            config={"configurable": {"session_id": session_id}}
    ):
        response_parts.append(chunk)
        chat_history[-1]["content"] = "".join(response_parts)
        yield chat_history, ""


with gr.Blocks(title="RAG Sohbet Asistanı") as demo:
    gr.Markdown("# RAG Sohbet Asistanı")

    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=["Türkçe", "İngilizce", "Almanca", "Fransızca", "İspanyolca"],
            value="Türkçe",
            label="Yanıt Dili"
        )

    chatbot = gr.Chatbot(height=500, type="messages")
    msg = gr.Textbox(placeholder="Sorunuzu buraya yazın...", label="Mesajınız")
    clear = gr.Button("Sohbeti Temizle")

    # Stream için generator fonksiyon
    msg.submit(respond, [msg, chatbot, language_dropdown], [chatbot, msg])
    clear.click(lambda: [], None, chatbot, queue=False)



demo.queue()
demo.launch(share=False)
