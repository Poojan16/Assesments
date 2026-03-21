import asyncio
import json
from aio_pika import connect, IncomingMessage,connect_robust, Message
from emailLogic import *
from fastapi import BackgroundTasks

async def Producer(email_schmea: EmailSchema1):
    connection = await connect_robust("amqp://guest:guest@localhost/")

    async with connection:
        channel = await connection.channel()

        queue_name = "email_queue"
        await channel.declare_queue(queue_name, durable=True)
        
        email_data = {
            'emails': email_schmea.emails,
            'subject': email_schmea.subject,
            'body': email_schmea.body
        }
        

        message_body = json.dumps(email_data).encode()
        message = Message(message_body)

        await channel.default_exchange.publish(
            message,
            routing_key=queue_name,
        )

        print(f" [x] Sent '{message_body.decode()}'")

async def on_message_received(message: IncomingMessage):
    """Callback function called when a message is received."""
    async with message.process():
        message_data = json.loads(message.body.decode())
        email_schema = EmailSchema1(emails=message_data['emails'], subject=message_data['subject'], body=message_data['body'])
        try:
            await send_email_endpoint(email_schema, BackgroundTasks())
        except Exception as e:
            if("550, '5.4.5 Daily user sending limit exceeded" in str(e)):
                logging.warning("Daily SMTP rate limit exceeded, waiting for reset")
                await asyncio.sleep(84600)
            if("550, '5.7.1 Message too large" in str(e)):
                logging.warning("Message too large, waiting for reset")
                await asyncio.sleep(5)
            print(f"Error sending email: {e}")
            await asyncio.sleep(5)
        await asyncio.sleep(1) 
        print(f" [x] Done processing task id: {message_data['emails']}")

async def Consumer():
    connection = await connect("amqp://guest:guest@localhost/")

    async with connection:
        channel = await connection.channel()
        
        queue_name = "email_queue"
        queue = await channel.declare_queue(queue_name, durable=True)
        
        await channel.set_qos(prefetch_count=1)

        await queue.consume(on_message_received)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        await asyncio.Future() 

