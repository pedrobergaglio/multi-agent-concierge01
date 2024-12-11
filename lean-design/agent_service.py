from message_broker import MessageBroker
import json

class AgentService:
    def __init__(self):
        self.message_broker = MessageBroker()
    
    def process_task(self, ch, method, properties, body):
        """Procesa una tarea del agente."""
        task = json.loads(body)
        print(f"Procesando tarea: {task}")

        # Simula procesamiento
        result = {"status": "success", "message": f"Task completed by {task['agent']}"}
        print(f"Tarea completada: {result}")

        # Aquí podrías enviar el resultado a un contexto o devolverlo a un sistema central.

    def start(self):
        """Comienza a escuchar la cola de tareas."""
        print("Agente escuchando tareas...")
        self.message_broker.consume("tasks", self.process_task)

if __name__ == "__main__":
    agent = AgentService()
    agent.start()
