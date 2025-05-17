"""E-posta gönderme araçları."""
import smtplib
from email.message import EmailMessage
from agents import function_tool
from ..config import SMTP_HOST, SMTP_PORT, EMAIL_FROM, EMAIL_TO


@function_tool
def send_email(body: str):
    """Düz metin e-posta gönderir (agent-as-tools yaklaşımı için)"""
    msg = EmailMessage()
    msg.set_content(body)

    msg['Subject'] = 'Test E-postası'
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
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
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.send_message(msg)
        print(f'HTML e-posta gönderildi! Konu: {subject}')
        return {"status": "success"}
    except Exception as e:
        print(f'Bir hata oluştu: {e}')
        return {"status": "failed", "error": str(e)}