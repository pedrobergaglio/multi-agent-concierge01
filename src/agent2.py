import asyncio

from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step


# create a dummy workflow
class Agent2Workflow(Workflow):
    """A dummy workflow with only one step sending back the input given."""

    @step()
    async def run_step(self, ev: StartEvent) -> StopEvent:
        message = str(ev.get("message", ""))
        return StopEvent(result=f"Message received by Agent2: {message}")


agent2_workflow = Agent2Workflow()


async def main():
    print(await agent2_workflow.run(message="Hello!"))


if __name__ == "__main__":
    asyncio.run(main())