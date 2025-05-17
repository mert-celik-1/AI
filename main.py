from dotenv import load_dotenv
import os
from agents import Agent, Runner, trace, function_tool
import asyncio
import smtplib
from email.message import EmailMessage

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

# ----- AJANLARIN TANIMI -----

# Satış ajanları
instructions1 = "You are a sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write professional, serious cold emails."

instructions2 = "You are a humorous, engaging sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write witty, engaging cold emails that are likely to get a response."

instructions3 = "You are a busy sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write concise, to the point cold emails."

sales_agent1 = Agent(
    name="Professional Sales Agent",
    instructions=instructions1,
    model="gpt-4o-mini",
)

sales_agent2 = Agent(
    name="Engaging Sales Agent",
    instructions=instructions2,
    model="gpt-4o-mini",
)

sales_agent3 = Agent(
    name="Busy Sales Agent",
    instructions=instructions3,
    model="gpt-4o-mini",
)

# E-posta formatları ile ilgili ajanlar
subject_instructions = "You can write a subject for a cold sales email. \
You are given a message and you need to write a subject for an email that is likely to get a response."

html_instructions = "You can convert a text email body to an HTML email body. \
You are given a text email body which might have some markdown \
and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

subject_writer = Agent(
    name="Email subject writer",
    instructions=subject_instructions,
    model="gpt-4o-mini"
)

html_converter = Agent(
    name="HTML email body converter",
    instructions=html_instructions,
    model="gpt-4o-mini"
)


# ----- FONKSİYON ARAÇLARI -----

@function_tool
def send_email(body: str):
    """Düz metin e-posta gönderir (agent-as-tools yaklaşımı için)"""
    msg = EmailMessage()
    msg.set_content(body)

    msg['Subject'] = 'Test E-postası'
    msg['From'] = 'gonderen@example.com'
    msg['To'] = 'mertcelik2399@gmail.com'

    try:
        with smtplib.SMTP('localhost', 1025) as server:
            server.send_message(msg)
        print(f'Düz metin e-posta gönderildi: {body[:100]}...')
        return {"status": "success"}
    except Exception as e:
        print(f'Bir hata oluştu: {e}')
        return {"status": "failed", "error": str(e)}


@function_tool
def send_html_email(subject: str, html_body: str) -> dict:
    """HTML e-posta gönderir (handoff yaklaşımı için)"""
    msg = EmailMessage()
    msg.set_content(html_body, subtype='html')

    msg['Subject'] = subject
    msg['From'] = 'gonderen@example.com'
    msg['To'] = 'mertcelik2399@gmail.com'

    try:
        with smtplib.SMTP('localhost', 1025) as server:
            server.send_message(msg)
        print(f'HTML e-posta gönderildi! Konu: {subject}')
        return {"status": "success"}
    except Exception as e:
        print(f'Bir hata oluştu: {e}')
        return {"status": "failed", "error": str(e)}


# ----- ARAÇLARIN TANIMLANMASI -----

description = "Write a cold sales email"
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

subject_tool = subject_writer.as_tool(tool_name="subject_writer",
                                      tool_description="Write a subject for a cold sales email")
html_tool = html_converter.as_tool(tool_name="html_converter",
                                   tool_description="Convert a text email body to an HTML email body")

# Araç setleri
sales_tools = [tool1, tool2, tool3]
email_format_tools = [subject_tool, html_tool, send_html_email]
direct_email_tools = sales_tools + [send_email]  # Agent-as-tools yaklaşımı için

# ----- HANDOFF İÇİN EMAİLER AGENT -----

emailer_instructions = "You are an email formatter and sender. You receive the body of an email to be sent. \
You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. \
Finally, you use the send_html_email tool to send the email with the subject and HTML body."

emailer_agent = Agent(
    name="Emailer Manager",
    instructions=emailer_instructions,
    tools=email_format_tools,
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it"
)

# ----- İKİ FARKLI SALES MANAGER -----

# 1. Agent-as-tools yaklaşımı (kontrol geri döner)
direct_manager_instructions = "You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. \
You never generate sales emails yourself; you always use the tools. \
You try all 3 sales_agent tools once before choosing the best one. \
You pick the single best email and use the send_email tool to send the best email (and only the best email) to the user."

direct_sales_manager = Agent(
    name="Direct Sales Manager",
    instructions=direct_manager_instructions,
    tools=direct_email_tools,  # send_email aracını içerir
    model="gpt-4o-mini"
)

# 2. Handoff yaklaşımı (kontrol devredilir)
handoff_manager_instructions = "You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. \
You never generate sales emails yourself; you always use the tools. \
You try all 3 sales agent tools at least once before choosing the best one. \
You can use the tools multiple times if you're not satisfied with the results from the first try. \
You select the single best email using your own judgement of which email will be most effective. \
After picking the email, you handoff to the Email Manager agent to format and send the email."

handoff_sales_manager = Agent(
    name="Handoff Sales Manager",
    instructions=handoff_manager_instructions,
    tools=sales_tools,  # Sadece e-posta oluşturma araçları
    handoffs=[emailer_agent],  # Emailer agent'a handoff
    model="gpt-4o-mini"
)


# ----- ÇALIŞTIRMA FONKSİYONLARI -----

async def run_direct_approach():
    """Agent-as-tools yaklaşımını çalıştırır (kontrol geri döner)"""
    message = "Send a cold sales email addressed to 'Dear CEO'"
    with trace('Direct Sales Manager (Agent-as-Tools)'):
        result = await Runner.run(direct_sales_manager, message)
        print("\n--- DIRECT APPROACH RESULT ---")
        print(result)


async def run_handoff_approach():
    """Handoff yaklaşımını çalıştırır (kontrol devredilir)"""
    message = "Send out a cold sales email addressed to Dear CEO from Alice"
    with trace('Handoff Sales Manager'):
        result = await Runner.run(handoff_sales_manager, message)
        print("\n--- HANDOFF APPROACH RESULT ---")
        print(result)


# ----- MAIN FONKSİYON -----

async def main():

    approach = "both"

    if approach == "direct" or approach == "both":
        await run_direct_approach()

    if approach == "handoff" or approach == "both":
        await run_handoff_approach()


if __name__ == "__main__":
    # python -m aiosmtpd -n -l localhost:1025
    asyncio.run(main())