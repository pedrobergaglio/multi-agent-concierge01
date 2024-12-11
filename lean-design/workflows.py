from context_service import ContextService
from message_broker import MessageBroker

class Workflow:
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.ctx_service = ContextService()
        self.message_broker = MessageBroker()
    
    async def run(self):
        # Obtener contexto inicial
        ctx = self.ctx_service.get_context(self.workflow_id)
        print("Contexto inicial:", ctx)

        # Ejecutar tarea con agente 1
        task = {"agent": "agent1", "payload": {"message": "Hello from Workflow"}}
        self.message_broker.publish("tasks", task)
        print("Tarea publicada para agente 1")

        # Simular espera de resultado (en un sistema real, usaría un mecanismo de retorno)
        result = {"status": "completed", "data": "Resultado de agente 1"}
        self.ctx_service.update_context(self.workflow_id, result)
        print("Contexto actualizado:", result)

if __name__ == "__main__":
    import asyncio
    workflow = Workflow("workflow1")
    asyncio.run(workflow.run())
