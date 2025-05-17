"""Handoff + Guardrails"""
import asyncio
from agents import Runner, trace
from .guardrails.name_check import guardrail_against_name
from .my_agents.email_agents import create_email_format_tools
from .my_agents.manager_agents import create_emailer_agent, create_handoff_guardrail_sales_manager
from .my_agents.sales_agents import create_sales_tools
from .tools.email_tools import send_html_email


async def run_guardrail_approach():
    """Handoff + Guardrail çalıştırır"""

    # Araçları oluştur
    sales_tools = create_sales_tools()
    email_format_tools = create_email_format_tools() + [send_html_email]


    # Ajanları oluştur
    emailer_agent = create_emailer_agent(email_format_tools)
    handoff_guardrail_sales_manager = create_handoff_guardrail_sales_manager(sales_tools,[emailer_agent],[guardrail_against_name])

    with trace('Handoff + Guardrail Sales Manager'):
        message = "Send out a cold sales email addressed to Dear CEO from Alice"
        result = await Runner.run(handoff_guardrail_sales_manager, message)
        print("\n--- HANDOFF APPROACH RESULT ---")
        print(result)


if __name__ == "__main__":
    asyncio.run(run_guardrail_approach())