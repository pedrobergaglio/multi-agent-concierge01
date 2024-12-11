


from flask import Flask, request, jsonify
from colorama import Fore, Style
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI
import asyncio
import threading

# Global queue to handle incoming user inputs asynchronously
user_input = asyncio.Queue()

app = Flask(__name__)

class JokeEvent(Event):
    joke: str

class JokeFlow(Workflow):
    llm = OpenAI(model="gpt-4o-mini", temperature=0.4)

    @step()  # Define the start and stop events explicitly
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        print(Fore.MAGENTA + 'Waiting for user input...' + Style.RESET_ALL)

        # Wait for user input from the API request
        user_msg_str = await user_input.get()

        prompt = f"Write your best joke about {user_msg_str}."
        response = await self.llm.acomplete(prompt)

        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        return JokeEvent(joke=str(response))

    @step()  # Link the JokeEvent to the StopEvent
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        question = await user_input.get()

        prompt = f"Answer the following question: '{question}' shortly, about the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))

# Dedicated event loop for asyncio tasks
loop = asyncio.new_event_loop()

async def run_workflow():
    # Create and run the workflow instance
    workflow = JokeFlow(timeout=60, verbose=True)
    result = await workflow.run()
    print('Workflow result:', result)

# Flask endpoint to receive user input
@app.route('/user-input', methods=['POST'])
def handle_user_input():
    data = request.json
    input_value = data.get('input', '')

    if input_value:
        # Put the input into the queue for the workflow to process using the dedicated event loop
        asyncio.run_coroutine_threadsafe(user_input.put(input_value), loop)
        return jsonify({'status': 'received', 'input': input_value})
    else:
        return jsonify({'status': 'error', 'message': 'No input provided'}), 400

if __name__ == "__main__":
    # Run the event loop in a separate thread
    threading.Thread(target=lambda: loop.run_forever()).start()

    # Run the workflow in the event loop
    asyncio.run_coroutine_threadsafe(run_workflow(), loop)

    # Start the Flask server
    app.run(port=8080)