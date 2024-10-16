from typing import List

class LlmMessage():
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "type": "string", "content": self.content}

class LlmToolSchemaAtribute():
    def __init__(self, attribute: str, type: str, description: str):
        self.attribute = attribute
        self.type = type
        self.description = description

    def to_dict(self):
        return {f'{self.attribute}': {"type": self.type, "description": self.description}}

# Receive any amount of LlmToolSchemaAtribute objects and return a dictionary
class LlmToolSchema():
    def __init__(self, atributes: List[LlmToolSchemaAtribute]):
        self._attributes = atributes
    
    def to_dict(self):
        return {attribute.attribute: {"type": attribute.type, "description": attribute.description} for attribute in self._attributes}



