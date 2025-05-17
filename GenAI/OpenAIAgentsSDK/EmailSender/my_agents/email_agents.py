"""E-posta formatlaması yapan ajanlar."""
from agents import Agent
from ..config import MODEL


def create_email_format_agents():
    """E-posta konu ve HTML dönüşüm ajanlarını oluşturur."""

    subject_instructions = """You can write a subject for a cold sales email. 
    You are given a message and you need to write a subject for an email that is likely to get a response."""

    html_instructions = """You can convert a text email body to an HTML email body. 
    You are given a text email body which might have some markdown 
    and you need to convert it to an HTML email body with simple, clear, compelling layout and design."""

    subject_writer = Agent(
        name="Email subject writer",
        instructions=subject_instructions,
        model=MODEL
    )

    html_converter = Agent(
        name="HTML email body converter",
        instructions=html_instructions,
        model=MODEL
    )

    return subject_writer, html_converter


def create_email_format_tools():
    """E-posta format araçlarını döndürür."""
    subject_writer, html_converter = create_email_format_agents()

    subject_tool = subject_writer.as_tool(
        tool_name="subject_writer",
        tool_description="Write a subject for a cold sales email"
    )

    html_tool = html_converter.as_tool(
        tool_name="html_converter",
        tool_description="Convert a text email body to an HTML email body"
    )

    return [subject_tool, html_tool]