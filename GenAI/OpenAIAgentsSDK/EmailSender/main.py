"""Ana program."""
import asyncio
from .run_direct import run_direct_approach
from .run_handoff import run_handoff_approach
from .run_guardrail import run_guardrail_approach


async def main():
    approach = "guardrail"

    if approach == "direct" or approach == "all":
        print("\n=== Running Direct Approach (Agent-as-Tools) ===\n")
        await run_direct_approach()

    if approach == "handoff" or approach == "all":
        print("\n=== Running Handoff Approach ===\n")
        await run_handoff_approach()

    if approach == "guardrail" or approach == "all":
        print("\n=== Running Guardrail Approach ===\n")
        await run_guardrail_approach()


if __name__ == "__main__":
    # python -m aiosmtpd -n -l localhost:1025
    asyncio.run(main())