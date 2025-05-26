from datetime import datetime

class DataPoint:
    def __init__(self, timestamp: str, value: float):
        self.timestamp = datetime.fromisoformat(timestamp)
        self.value = value

    def get_timestamp(self):
        return self.timestamp

    def set_timestamp(self, timestamp: str):
        self.timestamp = datetime.fromisoformat(timestamp)

    def get_value(self):
        return self.value

    def set_value(self, value: float):
        self.value = value

    def __str__(self):
        return f"({self.timestamp}, {self.value})"