from api.context_client import get_context, set_context
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
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from enum import Enum
from typing import Optional, List, Callable
from colorama import Fore, Back, Style
from src.concierge_agent import ConciergeAgent
from src.context_service import ContextService
from src.context_service import context_service
from events import Events
from functions import Functions
from concierge_workflow import ConciergeWorkflow

async def main():
    c = ConciergeWorkflow(timeout=1200, verbose=True)
    #set_context({"data":{"transfer_money_agent": "example"}})
    result = await c.run()
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
