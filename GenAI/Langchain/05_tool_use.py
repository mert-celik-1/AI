from dotenv import load_dotenv
import os
import requests
import gradio as gr
import re
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

model = init_chat_model("gpt-4o-mini", model_provider="openai", streaming=True)

pdf_paths = ["attention.pdf", "avelandia.pdf"]

pdf_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    pdf_docs.extend(loader.load())

web_loader = WebBaseLoader(
    "https://cbarkinozer.medium.com/transformat%C3%B6rler-t%C3%BCm-i%CC%87htiyac%C4%B1n%C4%B1z-olan-dikkat-ee4ff66723b1")
web_docs = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
chunks = text_splitter.split_documents(pdf_docs + web_docs)

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


def clean_text(text):
    return text.encode('utf-8', errors='ignore').decode('utf-8')


@tool
def get_weather(location: str) -> str:
    """Belirli bir konumun güncel hava durumunu API'den çeker."""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return "API anahtarı bulunamadı. OPENWEATHER_API_KEY çevre değişkenini ayarlayın."

        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
            "lang": "tr"
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        result = f"{location} için hava durumu:\n"
        result += f"Durum: {weather_description}\n"
        result += f"Sıcaklık: {temperature}°C (Hissedilen: {feels_like}°C)\n"
        result += f"Nem: %{humidity}\n"
        result += f"Rüzgar Hızı: {wind_speed} m/s"

        return result

    except requests.exceptions.HTTPError as http_err:
        if 'response' in locals() and response.status_code == 404:
            return f"'{location}' konumu bulunamadı. Lütfen geçerli bir şehir veya ülke adı girin."
        else:
            return f"HTTP hatası: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"İstek hatası: {req_err}"
    except KeyError as key_err:
        return f"API yanıtında beklenen veri bulunamadı: {key_err}"
    except Exception as e:
        return f"Beklenmeyen hata: {str(e)}"


tools = [get_weather]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """Akıllı bir asistansın. Kullanıcı sorularını yanıtlamak ve çeşitli görevleri yerine getirmek için araçları kullanabilirsin.

    Elinde şu araçlar var:
    1. Hava durumu bilgisi alma

    Araçları kullanırken doğru parametreleri sağladığından emin ol. Kullanıcının isteğini anla ve en uygun aracı seç.

    Araç kullanımı gerekmeyen sorular için kendi bilgilerinle yanıt ver.

    KULLANICI TARAFINDAN BELİRTİLEN DİL: {language}. Bu dilde yanıt vermeyi unutma."""),
    ("human", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(model, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


class CriticResult(BaseModel):
    is_hallucination: bool = Field(description="Yanıtın bir halüsinasyon içerip içermediği")
    score: int = Field(description="Yanıtın doğruluk puanı (1-10 arası)")
    issues: Optional[List[str]] = Field(description="Tespit edilen sorunlar listesi")
    corrected_response: Optional[str] = Field(description="Düzeltilmiş yanıt")
    reasoning: str = Field(description="Değerlendirme gerekçesi")


@tool
def analyze_response(response_text: str) -> str:
    """Verilen yanıtı analiz eder. Bu sadece critic agent'ın OpenAI fonksiyonları için gerekli bir fonksiyondur."""
    return "Analiz tamamlandı."


critic_tools = [analyze_response]

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen bir yanıt değerlendirme uzmanısın. Kullanıcı sorusuna ve AI tarafından verilen cevaba dayanarak, 
    cevabın doğruluğunu değerlendir ve halüsinasyon içerip içermediğini belirle.

    Değerlendirmeni aşağıdaki formatta yap:

    IS_HALLUCINATION: [true/false] - Yanıt halüsinasyon içeriyor mu?
    SCORE: [1-10] - Yanıtın doğruluk puanı
    ISSUES: [liste] - Tespit edilen sorunlar
    REASONING: [açıklama] - Değerlendirme gerekçen
    CORRECTED_RESPONSE: [düzeltilmiş yanıt] - Gerekiyorsa yanıtı düzelt

    Değerlendirme yaparken:
    1. Yanıtın, kullanıcı sorusuyla alakalı olup olmadığını kontrol et
    2. Yanıttaki bilgilerin doğruluk derecesini değerlendir
    3. Eğer halüsinasyon (uydurma bilgi) tespit edersen, hangi kısımların sorunlu olduğunu belirt
    4. Sorunlu yanıtları düzelt veya belirsizliği azalt
    5. Yanıta 1-10 arası bir doğruluk puanı ver
    """),
    ("human", """Kullanıcı Sorusu: {question}

    AI Yanıtı: {ai_response}

    Lütfen bu yanıtı değerlendir."""),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

critic_model = init_chat_model("gpt-4o-mini", model_provider="openai", streaming=False)
critic_agent = create_openai_functions_agent(critic_model, critic_tools, critic_prompt)
critic_executor = AgentExecutor(agent=critic_agent, tools=critic_tools, verbose=True)


def evaluate_response(question, ai_response):
    try:
        evaluation_result = critic_executor.invoke({
            "question": question,
            "ai_response": ai_response
        })

        response_text = evaluation_result["output"]

        is_hallucination = "IS_HALLUCINATION: true" in response_text.upper()

        score_match = re.search(r"SCORE:\s*(\d+)", response_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 5

        issues_section = re.search(r"ISSUES:(.*?)(?:REASONING:|CORRECTED_RESPONSE:|$)",
                                   response_text, re.IGNORECASE | re.DOTALL)
        issues = []
        if issues_section:
            issues_text = issues_section.group(1).strip()
            issues = [issue.strip() for issue in issues_text.split("\n") if issue.strip()]

        corrected_section = re.search(r"CORRECTED_RESPONSE:(.*?)(?:$)",
                                      response_text, re.IGNORECASE | re.DOTALL)
        corrected_response = corrected_section.group(1).strip() if corrected_section else None

        reasoning_section = re.search(r"REASONING:(.*?)(?:CORRECTED_RESPONSE:|$)",
                                      response_text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_section.group(1).strip() if reasoning_section else "Değerlendirme yapılamadı."

        result = CriticResult(
            is_hallucination=is_hallucination,
            score=score,
            issues=issues,
            corrected_response=corrected_response,
            reasoning=reasoning
        )

        return result

    except Exception as e:
        print(f"Critic Agent hatası: {str(e)}")
        return CriticResult(
            is_hallucination=False,
            score=5,
            issues=["Değerlendirme sırasında teknik bir hata oluştu."],
            corrected_response=None,
            reasoning="Değerlendirme yapılamadı."
        )


def respond(message, chat_history, language, show_debug=False):
    session_id = "gradio_session"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})

    yield chat_history, ""

    try:
        print("\n=== MAIN AGENT EXECUTION ===")
        response = agent_executor.invoke({
            "question": message,
            "language": language
        })

        initial_response = response["output"]
        print(f"\nİlk yanıt: {initial_response}\n")

        print("\n=== CRITIC AGENT EXECUTION ===")
        evaluation = evaluate_response(message, initial_response)
        print(f"\nDeğerlendirme sonucu: {evaluation}\n")

        if evaluation.is_hallucination and evaluation.corrected_response:
            final_response = f"{evaluation.corrected_response}"

            if show_debug:
                final_response += f"\n\n---\n*Bu yanıt düzeltildi.*\n"
                final_response += f"*Güvenilirlik puanı: {evaluation.score}/10*\n"
                final_response += f"*Sorunlar: {', '.join(evaluation.issues)}*\n"
                final_response += f"*Gerekçe: {evaluation.reasoning}*"
        else:
            final_response = initial_response

            if show_debug:
                final_response += f"\n\n---\n*Güvenilirlik puanı: {evaluation.score}/10*\n"
                if evaluation.reasoning:
                    final_response += f"*Değerlendirme: {evaluation.reasoning}*"

        if "data:image/png;base64," in final_response:
            image_parts = final_response.split("data:image/png;base64,")
            text_part = image_parts[0]
            image_data = image_parts[1].split('"')[0] if '"' in image_parts[1] else image_parts[1]

            final_response = f"{text_part}\n\n<img src='data:image/png;base64,{image_data}' alt='Oluşturulan Grafik'>"

        chat_history[-1]["content"] = final_response
        yield chat_history, ""

    except Exception as e:
        error_message = f"Üzgünüm, bir hata oluştu: {str(e)}"
        chat_history[-1]["content"] = error_message
        yield chat_history, ""


with gr.Blocks(title="Agentic RAG Asistanı") as demo:
    gr.Markdown("# Agentic RAG Asistanı")
    gr.Markdown("""Bu asistan, metin tabanlı sorulara cevap vermenin yanı sıra aşağıdaki araçları kullanabilir:

    - 🌤️ Hava durumu bilgisi

    Asistan ayrıca yanıtlarını kontrol eden bir değerlendirme sistemi içerir ve halüsinasyonları önlemeye çalışır.
    """)

    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=["Türkçe", "İngilizce", "Almanca", "Fransızca", "İspanyolca"],
            value="Türkçe",
            label="Yanıt Dili"
        )

        show_debug = gr.Checkbox(
            label="Değerlendirme Bilgilerini Göster",
            value=False
        )

    chatbot = gr.Chatbot(height=500, type="messages")
    msg = gr.Textbox(placeholder="Sorunuzu buraya yazın veya bir araç kullanmak istediğinizi belirtin...",
                     label="Mesajınız")
    clear = gr.Button("Sohbeti Temizle")

    msg.submit(respond, [msg, chatbot, language_dropdown, show_debug], [chatbot, msg])
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=False)