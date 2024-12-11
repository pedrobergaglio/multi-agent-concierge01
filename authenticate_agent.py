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

from src.concierge_agent import ConciergeAgent

#from llama_index.llms.anthropic import Anthropic
""" from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from enum import Enum
from typing import Optional, List, Callable
#from llama_index.utils.workflow import draw_all_possible_flows
from colorama import Fore, Back, Style

from concierge_agent import ConciergeAgent
 """
#from functions import chat_history
#from functions import Functions
from events import Events
from src.context_service import ContextService
from src.context_service import context_service
from api.context_client import get_context, set_context

async def authenticate(self, ctx:Context, ev: StartEvent) -> StopEvent: #Events.ConciergeEvent:

    ctx = get_context()
    print("Running authenticate_workflow", type(ctx))

    ev = ctx.data["event"]

    if "authentication_agent" not in ctx.data:
        def store_username(username: str) -> None:
            ctx.data["user"]["username"] = username

        def login(password: str) -> None:
            session_token = "output_of_login_function_goes_here"
            ctx.data["user"]["session_token"] = session_token

        def is_authenticated() -> bool:
            return ctx.data["user"]["session_token"] is not None

        system_prompt = ("""
            You are a helpful assistant that is authenticating a user.
            Your task is to get a valid session token stored in the user state.
            To do this, the user must supply you with a username and a valid password. You can ask them to supply these.
            If the user supplies a username and password, call the tool "login" to log them in.
            Once you've called the login tool successfully, call the tool named "done" to signal that you are done. Do this before you respond.
            If the user asks to do anything other than authenticate, call the tool "need_help" to signal some other agent should help.
        """)

        ctx.data["authentication_agent"] = ConciergeAgent(
            name="Authentication Agent",
            parent=self,
            tools=[store_username, login, is_authenticated],
            system_prompt=system_prompt,
            #trigger_event=Events.AuthenticateEvent
        )

    result = ctx.data["authentication_agent"].handle_event(ev)
    ContextService.set_context(ctx)
    return result

