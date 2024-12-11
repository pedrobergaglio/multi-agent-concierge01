import json
from jokes import *
from flask import Flask, request
import asyncio

# api/services.py

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)

# In-memory store for user workflow states
user_workflow_states = {}

workflow = JokeFlow(timeout=1200, verbose=True)

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def handle_user_message_route():
    user_id = request.json.get('from')
    message = request.json.get('message')

    response = asyncio.run(handle_user_message(user_id, message))
    return response

async def handle_user_message(user_id, message):

    result = await workflow.run(message=message, event=IntentionEvent)
    print(result)
    result = await workflow.run(event=result['next_call'])
    print(result)

    return 'success'


if __name__ == "__main__":
    app.run(port=5001, debug=True)
