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


class ConciergeWorkflow(Workflow):

    @step
    async def initialize(self, ev: Events.InitializeEvent) -> Events.ConciergeEvent:

        ctx = get_context()

        print("Running concierge_workflow")
        print(ctx)

        ctx["user"] = {
            "username": "username",
            "session_token": None,
            "account_id": None,
            "account_balance": None,
        }
        ctx["success"] = None
        ctx["redirecting"] = None
        ctx["overall_request"] = None

        ctx["llm"] = OpenAI(model="gpt-4o-mini", temperature=0.4)
        # ctx["llm"] = Anthropic(model="claude-3-5-sonnet-20240620",temperature=0.4)
        # ctx["llm"] = Anthropic(model="claude-3-opus-20240229",temperature=0.4)

        set_context(ctx)

        return Events.ConciergeEvent()

    @step
    async def concierge(self, ev: Events.ConciergeEvent | StartEvent) -> Events.InitializeEvent |  StopEvent | Events.OrchestratorEvent:
        ctx = get_context()

        if "user" not in ctx:
            return Events.InitializeEvent()

        if "concierge" not in ctx:
            system_prompt = ("""
                You are a helpful assistant that is helping a user navigate a financial system.
                Your job is to ask the user questions to figure out what they want to do, and give them the available things they can do.
                That includes
                * looking up a stock price            
                * authenticating the user
                * checking an account balance
                * transferring money between accounts
                You should start by listing the things you can help them do.            
            """)

            agent_worker = FunctionCallingAgentWorker.from_tools(
                tools=[],
                llm=ctx["llm"],
                allow_parallel_tool_calls=False,
                system_prompt=system_prompt
            )
            ctx["concierge"] = agent_worker.as_agent()

        concierge = ctx["concierge"]
        if ctx["overall_request"] is not None:
            last_request = ctx["overall_request"]
            ctx["overall_request"] = None
            set_context(ctx)
            return Events.OrchestratorEvent(request=last_request)
        elif ev.just_completed is not None:
            response = concierge.chat(f"FYI, the user has just completed the task: {ev.just_completed}")
        elif ev.need_help:
            set_context(ctx)
            return Events.OrchestratorEvent(request=ev.request)
        else:
            response = concierge.chat("Hello!")

        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)
        user_msg_str = input("> ").strip()
        set_context(ctx)
        return Events.OrchestratorEvent(request=user_msg_str)
