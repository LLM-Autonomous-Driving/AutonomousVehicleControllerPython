import uuid

from confluent_kafka import Producer


class Publisher:
    def __init__(self, server: str):
        self.publisher = Producer({'bootstrap.servers': server})

    def publish(self, topic: str, key: uuid, value: bytes):
        try:
            key = str(key)

            self.publisher.produce(topic, key=key, value=value)
            self.publisher.flush()
        except Exception as e:
            print(f"Error publishing to {topic}: {e}")

        return f"Published to {topic} with key {key} and value {value}"


