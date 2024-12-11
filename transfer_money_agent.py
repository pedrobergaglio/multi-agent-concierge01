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
from concierge_workflow import ConciergeAgent, Events, get_context, set_context, ConciergeWorkflow

@step(workflow = ConciergeWorkflow)
def transfer_money(self, ctx: Context, ev: Events.TransferMoneyEvent) -> Events.AccountBalanceEvent | Events.ConciergeEvent:
    #ctx = get_context()

    if "transfer_money_agent" not in ctx.data:
        def transfer_money(from_account_id: str, to_account_id: str, amount: int) -> None:
            return f"Transferred {amount} to account {to_account_id}"

        def balance_sufficient(account_id: str, amount: int) -> bool:
            return ctx.data["user"]['account_balance'] >= amount

        def has_balance() -> bool:
            return ctx.data["user"]["account_balance"] is not None and ctx.data["user"]["account_balance"] > 0

        def is_authenticated() -> bool:
            return ctx.data["user"]["session_token"] is not None

        def authenticate() -> None:
            ctx.data["redirecting"] = True
            ctx.data["overall_request"] = "Transfer money"
            self.send_event(Events.AuthenticateEvent(request="Authenticate"))

        def check_balance() -> None:
            ctx.data["redirecting"] = True
            ctx.data["overall_request"] = "Transfer money"
            self.send_event(Events.AccountBalanceEvent(request="Check balance"))

        system_prompt = ("""
            You are a helpful assistant that transfers money between accounts.
            The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
            If they aren't authenticated, tell them to authenticate first.
            The user must also have looked up their account balance already, which you can check with the has_balance tool.
            If they haven't already, tell them to look up their account balance first.
            Once you have transferred the money, you can call the tool named "done" to signal that you are done. Do this before you respond.
            If the user asks to do anything other than transfer money, call the tool "done" to signal some other agent should help.
        """)

        ctx.data["transfer_money_agent"] = ConciergeAgent(
            name="Transfer Money Agent",
            parent=self,
            tools=[transfer_money, balance_sufficient, has_balance, is_authenticated, authenticate, check_balance],
            system_prompt=system_prompt,
            trigger_event=Events.TransferMoneyEvent
        )

    result = ctx.data["transfer_money_agent"].handle_event(ev)
    #set_context(ctx)
    return result
