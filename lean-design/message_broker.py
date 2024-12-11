import pika
import json

class MessageBroker:
    def __init__(self, host='localhost'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host))
        self.channel = self.connection.channel()
    
    def publish(self, queue: str, message: dict):
        """Publica un mensaje en una cola."""
        self.channel.queue_declare(queue=queue)
        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            body=json.dumps(message)
        )
    
    def consume(self, queue: str, callback):
        """Escucha una cola y ejecuta un callback por mensaje."""
        self.channel.queue_declare(queue=queue)
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=callback,
            auto_ack=True
        )
        self.channel.start_consuming()
