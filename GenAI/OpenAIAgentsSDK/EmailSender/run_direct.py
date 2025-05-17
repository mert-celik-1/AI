"""Agent-as-tools yaklaşımını çalıştıran script."""
import asyncio
from agents import Runner, trace
from .my_agents.sales_agents import create_sales_tools
from .my_agents.manager_agents import create_direct_sales_manager
from .tools.email_tools import send_email


async def run_direct_approach():
    """Agent-as-tools yaklaşımını çalıştırır."""
    sales_tools = create_sales_tools()
    direct_email_tools = sales_tools + [send_email]

    direct_sales_manager = create_direct_sales_manager(direct_email_tools)

    message = "Send a cold sales email addressed to 'Dear CEO'"
    with trace('Direct Sales Manager (Agent-as-Tools)'):
        result = await Runner.run(direct_sales_manager, message)
        print("\n--- DIRECT APPROACH RESULT ---")
        print(result)


if __name__ == "__main__":
    asyncio.run(run_direct_approach())