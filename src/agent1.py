import asyncio

from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from dotenv import load_dotenv
load_dotenv()

from llama_index.core.workflow import (
    step, 
    Context, 
    Workflow, 
    Event, 
    StartEvent, 
    StopEvent
)

#from llama_index.llms.anthropic import Anthropic
""" from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from enum import Enum
from typing import Optional, List, Callable
#from llama_index.utils.workflow import draw_all_possible_flows
from colorama import Fore, Back, Style

from concierge_agent import ConciergeAgent
from events import Events """
#from functions import chat_history
#from functions import Functions
""" import sys
import os
sys.path.append(os.path.abspath('../'))
from workflows import ctx """

# create a dummy workflow
class Agent1Workflow(Workflow):
    """A dummy workflow with only one step sending back the input given."""

    @step()
    async def run_step(self, ev: StartEvent) -> StopEvent:
        message = str(ev.get("message", ""))
        return StopEvent(result=f"Message received: {message}")



agent1_workflow = Agent1Workflow()


async def main():
    print("Running agent1_workflow")
    print(await agent1_workflow.run(message="Hello!"))


if __name__ == "__main__":
    asyncio.run(main())