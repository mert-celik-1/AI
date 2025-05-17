"""Satış e-postası oluşturan ajanlar."""
from agents import Agent
from ..config import MODEL

def create_sales_agents():
    """Satış ajanlarını oluşturur ve döndürür."""

    instructions1 = """You are a sales agent working for ComplAI, 
    a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
    You write professional, serious cold emails."""

    instructions2 = """You are a humorous, engaging sales agent working for ComplAI, 
    a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
    You write witty, engaging cold emails that are likely to get a response."""

    instructions3 = """You are a busy sales agent working for ComplAI, 
    a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
    You write concise, to the point cold emails."""

    sales_agent1 = Agent(
        name="Professional Sales Agent",
        instructions=instructions1,
        model=MODEL,
    )

    sales_agent2 = Agent(
        name="Engaging Sales Agent",
        instructions=instructions2,
        model=MODEL,
    )

    sales_agent3 = Agent(
        name="Busy Sales Agent",
        instructions=instructions3,
        model=MODEL,
    )

    return sales_agent1, sales_agent2, sales_agent3


def create_sales_tools():
    """Satış ajanlarını araç olarak döndürür."""
    sales_agent1, sales_agent2, sales_agent3 = create_sales_agents()

    description = "Write a cold sales email"
    tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
    tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
    tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

    return [tool1, tool2, tool3]