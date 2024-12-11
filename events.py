from typing import Union
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
#from llama_index.llms.anthropic import Anthropic
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from enum import Enum
from typing import Optional, List, Callable
#from llama_index.utils.workflow import draw_all_possible_flows, draw_most_recent_execution
from colorama import Fore, Back, Style
from llama_index.core.llms import ChatMessage, MessageRole


class Events:

    class InitializeEvent(Event):
        pass

    class ConciergeEvent(Event):
        request: Optional[str] = None
        just_completed: Optional[str] = None
        need_help: Optional[bool] = None

    class OrchestratorEvent(Event):
        request: str

    class StockLookupEvent(Event):
        request: str

    class AuthenticateEvent(Event):
        request: str

    class AccountBalanceEvent(Event):
        request: str

    class TransferMoneyEvent(Event):
        request: str


OrchestratorEvents = Union[
    Events.ConciergeEvent, 
    Events.AuthenticateEvent, 
    Events.StockLookupEvent, 
    Events.AccountBalanceEvent, 
    StopEvent,
    Events.TransferMoneyEvent
    ]