from requests import Session
from argparse import ArgumentParser
from urllib.parse import urljoin


class bcolors:
    HEADER = '\033[95m'  #]
    BLUE = '\033[94m'  #]
    CYAN = '\033[96m'  #]
    GREEN = '\033[92m' #]
    OKBLUE = '\033[94m'  #]
    OKCYAN = '\033[96m'  #]
    OKGREEN = '\033[92m' #]
    WARNING = '\033[93m' #]
    FAIL = '\033[91m'   #]
    ENDC = '\033[0m'    #]
    BOLD = '\033[1m'    #]
    UNDERLINE = '\033[4m' #]


class LLMChat:
    def __init__(self, host: str, prompt: str | None = None, model_name: str = 'Qwen'):
        self.session = Session()
        self.host_url = host
        if not prompt:
            self.history = []
        else:
            self.history = [{
                'role': 'system',
                'content': prompt
            }]
        self.model_name = model_name
    
    def send_message(self, message: str, history=None):
        if history is None:
            history = []
        res = self.session.post(
            urljoin(urljoin(self.host_url, 'models/'), self.model_name),
            json={
                'inputs': history,
                'parameters': {
                    'use_cache': False, 'max_new_tokens': 150, 'temperature': 0.7, 'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.2, 'num_beams': 4 #'presence_penalty': 0.3
                }
            }
        )
        res.raise_for_status()
        res = res.json()
        history.append({'role': 'assistant', 'content': res[0]['generated_text']})
        return res[0]['generated_text'], history

    def chat(self, message: str):
        response, self.history = self.send_message(message, self.history)
        return response

    def close(self):
        self.session.close()

