"""Yönetici ajanlar."""
from agents import Agent

from ..config import MODEL


def create_emailer_agent(email_format_tools):
    """E-posta formatlaması ve gönderimi yapan ajanı oluşturur."""

    emailer_instructions = """You are an email formatter and sender. You receive the body of an email to be sent. 
    You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. 
    Finally, you use the send_html_email tool to send the email with the subject and HTML body."""

    emailer_agent = Agent(
        name="Emailer Manager",
        instructions=emailer_instructions,
        tools=email_format_tools,
        model=MODEL,
        handoff_description="Convert an email to HTML and send it"
    )

    return emailer_agent


def create_direct_sales_manager(direct_email_tools):
    """Agent-as-tools yaklaşımını kullanan satış yöneticisini oluşturur."""

    direct_manager_instructions = """You are a sales manager working for ComplAI. Your task is to generate cold sales emails.

    IMPORTANT INSTRUCTIONS - FOLLOW EXACTLY:
    1. Use sales_agent1 EXACTLY ONCE
    2. Use sales_agent2 EXACTLY ONCE
    3. Use sales_agent3 EXACTLY ONCE
    4. Compare the three emails and select the single best one
    5. Use send_email EXACTLY ONCE to send only the best email
    
    DO NOT use any tool more than once under any circumstances.
    DO NOT generate emails yourself - only use the tools.
    DO NOT skip any of the 3 sales agents.
    """

    direct_sales_manager = Agent(
        name="Direct Sales Manager",
        instructions=direct_manager_instructions,
        tools=direct_email_tools,
        model=MODEL
    )

    return direct_sales_manager


def create_handoff_sales_manager(sales_tools, handoffs):
    """Handoff yaklaşımını kullanan satış yöneticisini oluşturur."""

    handoff_manager_instructions = """You are a sales manager working for ComplAI. Your task is to generate cold sales emails.

    IMPORTANT INSTRUCTIONS - FOLLOW EXACTLY:
    1. Use sales_agent1 EXACTLY ONCE
    2. Use sales_agent2 EXACTLY ONCE
    3. Use sales_agent3 EXACTLY ONCE
    4. Compare the three emails and select the single best one
    5. Handoff to the Email Manager with ONLY the best email
    
    DO NOT use any tool more than once under any circumstances.
    DO NOT generate emails yourself - only use the tools.
    DO NOT skip any of the 3 sales agents.
    NO MATTER how the emails look, DO NOT try to improve them by calling the tools again.
    """

    handoff_sales_manager = Agent(
        name="Handoff Sales Manager",
        instructions=handoff_manager_instructions,
        tools=sales_tools,
        handoffs=handoffs,
        model=MODEL
    )

    return handoff_sales_manager


def create_handoff_guardrail_sales_manager(sales_tools, handoffs,guardrails):
    """Handoff + Guardrail yaklaşımını kullanan satış yöneticisini oluşturur."""

    handoff_guardrail_manager_instructions = """You are a sales manager working for ComplAI. Your task is to generate cold sales emails.

    IMPORTANT INSTRUCTIONS - FOLLOW EXACTLY:
    1. Use sales_agent1 EXACTLY ONCE
    2. Use sales_agent2 EXACTLY ONCE
    3. Use sales_agent3 EXACTLY ONCE
    4. Compare the three emails and select the single best one
    5. Handoff to the Email Manager with ONLY the best email

    DO NOT use any tool more than once under any circumstances.
    DO NOT generate emails yourself - only use the tools.
    DO NOT skip any of the 3 sales agents.
    NO MATTER how the emails look, DO NOT try to improve them by calling the tools again.
    """

    careful_handoff_sales_manager = Agent(
        name="Handoff Sales Manager With Guardrails",
        instructions=handoff_guardrail_manager_instructions,
        tools=sales_tools,
        handoffs=handoffs,
        model=MODEL,
        input_guardrails=guardrails
    )



    return careful_handoff_sales_manager