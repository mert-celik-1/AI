from agents import Agent, input_guardrail, Runner, GuardrailFunctionOutput
from pydantic import BaseModel
from ..config import MODEL


class NameCheckOutput(BaseModel):
    is_name_in_message : bool
    name : str



guardrail_agent = Agent(
    name = "Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type = NameCheckOutput,
    model = MODEL
)

@input_guardrail
async def guardrail_against_name(ctx,agent,message):
    result = await Runner.run(guardrail_agent,message,context=ctx.context)
    print(result)
    is_name_in_message = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(output_info={"found_name":result.final_output},tripwire_triggered=is_name_in_message)



