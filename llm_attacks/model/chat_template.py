# Basic APIs of conversation/chat templates

class ChatTemplate():
    def __init__(self, name="", roles=['user', 'assistant']):
        self.name = name
        self.roles = roles

    def init_chat(self):
        return []

    def append_message(self, chat, role, message):
        chat.append({"role": role, "content": message})

    def apply_chat_template(self, chat):
        raise NotImplementedError()