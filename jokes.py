from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)

# `pip install llama-index-llms-openai` if you don't already have it
from llama_index.llms.openai import OpenAI


class AssistanceEvent(Event):
    None

class IntentionEvent(Event):
    None
""" 
class StartEvent(Event):
    message:str
    event:Event """


class JokeFlow(Workflow):

    @step(pass_context=True)
    async def header(self, ctx: Context, ev: StartEvent) -> AssistanceEvent | StopEvent | IntentionEvent:

        ctx.data['last_message'] = ev.get('message')

        if not ev.get('event'):
            # throw error or use hello event
            return IntentionEvent()

        if 'llm' not in ctx.data:
            print('added llm')
            ctx.data['llm'] = OpenAI(model="gpt-4o-mini", temperature=0.4)
        
        return ev.event()

    @step(pass_context=True)
    async def infer_intention(self, ctx: Context, ev: IntentionEvent) -> AssistanceEvent | StopEvent:

        last_message = ctx.data['last_message']
        ctx.data['intention'] = await ctx.data['llm'].acomplete('infer the user intention in one sentence in:' + last_message)
        return StopEvent(result={
            'next_call': AssistanceEvent,
            'response': str(ctx.data['intention'])
        })

    @step(pass_context=True)
    async def assist_intention(self, ctx: Context, ev: AssistanceEvent) -> StopEvent:

        intention = ctx.data['intention']

        prompt = f"this is the user intention, offer some things you can help with, keep the answer in one sentence: {intention}"
        response = await ctx.data['llm'].acomplete(prompt)
        ctx.data['response'] = response
        return StopEvent(result={
            'last_call': 'helper',
            'response': str(response)
        })

#draw_all_possible_flows(JokeFlow,filename="jokes_flows.html")

async def main():
    c = JokeFlow(timeout=1200, verbose=True)
    result = await c.run(message="i cant fix a python issue", event=IntentionEvent)
    print(result)
    result = await c.run(event=result['next_call'])
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())