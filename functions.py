

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

#from src.concierge_agent import ConciergeAgent
from events import Events
import requests

from src.context_service import ContextService
from src.authenticate_agent import AuthenticateAgent
#from transfer_money_agent import transfer_money
from authenticate_agent import authenticate

chat_history = []

class Functions:

    def emit_authenticate(ctx:Context) -> bool:
            print("__emitted: authenticate")
            #response = Functions.call_agent_via_api("authenticate_workflow")
            authenticate(ctx)
            #print(response)
            return True
    
    """ def emit_authenticate() -> bool:
            print("__emitted: authenticate")
            response = Functions.call_agent_via_api("authenticate_workflow")
            print(response)
            return True """

    def emit_agent1() -> bool:
        """
        Call this if the user wants to run the secret agent.
        """
        ctx = ContextService.get_context()
        
        print("__emitted: agent1")
        username = None# ctx.data["user"]["username"]
        response = Functions.call_agent_via_api("agent1_workflow", f"Hello from {username}!")
        print(response)
        return True
 
    def call_agent_via_api(agent_id: str, message: str = "") -> dict:
        url = "http://0.0.0.0:4501/deployments/QuickStart/tasks/run"
        payload = {
            "input": f"{{\"message\": \"{message}\"}}",
            "agent_id": agent_id
        }
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        return response.json()

   