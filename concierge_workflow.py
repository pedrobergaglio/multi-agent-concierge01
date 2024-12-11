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
from events import Events, OrchestratorEvents
from functions import Functions
from base_concierge_workflow import ConciergeWorkflow

class ConciergeWorkflow(ConciergeWorkflow):
    @step
    async def orchestrator(self, ev: Events.OrchestratorEvent) -> OrchestratorEvents:
        
        ctx = get_context()

        ctx['event'] = ev
        set_context(ctx)

        def emit_stock_lookup() -> bool:
            self.send_event(Events.StockLookupEvent(request=ev.request))
            return True

        def emit_account_balance() -> bool:
            self.send_event(Events.AccountBalanceEvent(request=ev.request))
            return True

        """ def emit_transfer_money() -> bool:
            self.send_event(Events.TransferMoneyEvent(request=ev.request))
            return True """

        def emit_concierge() -> bool:
            self.send_event(Events.ConciergeEvent(request=ev.request))
            return True
        
        def emit_authenticate() -> bool:
            self.send_event(Events.AuthenticateEvent(request=ev.request))
            return True

        def emit_stop() -> bool:
            self.send_event(StopEvent())
            return True

        tools = [
            FunctionTool.from_defaults(fn=emit_stock_lookup),
            FunctionTool.from_defaults(fn=emit_authenticate),
            FunctionTool.from_defaults(fn=emit_account_balance),
            #FunctionTool.from_defaults(fn=emit_transfer_money),
            FunctionTool.from_defaults(fn=emit_concierge),
            FunctionTool.from_defaults(fn=emit_stop),
        ]

        system_prompt = ("""
            You are on orchestration agent.
            Your job is to decide which agent to run based on the current state of the user and what they've asked to do. 
            You run an agent by calling the appropriate tool for that agent.
            You do not need to call more than one tool.
            You do not need to figure out dependencies between agents; the agents will handle that themselves.
                            
            If you did not call any tools, return the string "FAILED" without quotes and nothing else.
        """)

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=ctx["llm"],
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt
        )
        ctx["orchestrator"] = agent_worker.as_agent()

        orchestrator = ctx["orchestrator"]
        response = str(orchestrator.chat(ev.request))

        if response == "FAILED":
            set_context(ctx)
            return Events.OrchestratorEvent(request=ev.request)

        set_context(ctx)

    @step
    async def stock_lookup(self, ev: Events.StockLookupEvent) -> Events.ConciergeEvent:
        ctx =get_context()

        if "stock_lookup_agent" not in ctx:
            def lookup_stock_price(stock_symbol: str) -> str:
                return f"Symbol {stock_symbol} is currently trading at $100.00"

            def search_for_stock_symbol(str: str) -> str:
                return str.upper()

            system_prompt = ("""
                You are a helpful assistant that is looking up stock prices.
                The user may not know the stock symbol of the company they're interested in,
                so you can help them look it up by the name of the company.
                You can only look up stock symbols given to you by the search_for_stock_symbol tool, don't make them up. Trust the output of the search_for_stock_symbol tool even if it doesn't make sense to you.
                Once you have retrieved a stock price, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than look up a stock symbol or price, call the tool "need_help" to signal some other agent should help.
            """)

            ctx["stock_lookup_agent"] = ConciergeAgent(
                name="Stock Lookup Agent",
                parent=self,
                tools=[lookup_stock_price, search_for_stock_symbol],
                system_prompt=system_prompt,
                trigger_event=Events.StockLookupEvent
            )

        result = ctx["stock_lookup_agent"].handle_event(ev)
        set_context(ctx)
        return result

    @step
    def account_balance(self, ev: Events.AccountBalanceEvent) -> Events.ConciergeEvent:
        ctx =get_context()

        if "account_balance_agent" not in ctx:
            def get_account_id(account_name: str) -> str:
                account_id = "1234567890"
                ctx["user"]["account_id"] = account_id
                return f"Account id is {account_id}"

            def get_account_balance(account_id: str) -> str:
                ctx["user"]["account_balance"] = 1000
                return f"Account {account_id} has a balance of ${ctx['user']['account_balance']}"

            def is_authenticated() -> bool:
                return ctx["user"]["session_token"] is not None

            """ def authenticate() -> None:
                ctx["redirecting"] = True
                ctx["overall_request"] = "Check account balance"
                self.send_event(Events.AuthenticateEvent(request="Authenticate")) """

            system_prompt = ("""
                You are a helpful assistant that is looking up account balances.
                The user may not know the account ID of the account they're interested in,
                so you can help them look it up by the name of the account.
                The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
                If they aren't authenticated, call the "authenticate" tool to trigger the start of the authentication process; tell them you have done this.
                If they're trying to transfer money, they have to check their account balance first, which you can help with.
                Once you have supplied an account balance, you must call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than look up an account balance, call the tool "need_help" to signal some other agent should help.
            """)

            ctx["account_balance_agent"] = ConciergeAgent(
                name="Account Balance Agent",
                parent=self,
                tools=[get_account_id, get_account_balance, is_authenticated],
                system_prompt=system_prompt,
                trigger_event=Events.AccountBalanceEvent
            )

        result = ctx["account_balance_agent"].handle_event(ev)
        set_context(ctx)
        return result
    
    @step
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

    @step
    async def authenticate(self, ctx:Context, ev: Events.AuthenticateEvent) -> StopEvent: #Events.ConciergeEvent:

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
