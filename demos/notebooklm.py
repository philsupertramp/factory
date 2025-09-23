import re
import os
from pydub import AudioSegment
from requests import Session
from argparse import ArgumentParser
from urllib.parse import urljoin


class bcolors:
    HEADER = '\033[95m'  #]
    OKBLUE = '\033[94m'  #]
    OKCYAN = '\033[96m'  #]
    OKGREEN = '\033[92m' #]
    WARNING = '\033[93m' #]
    FAIL = '\033[91m'   #]
    ENDC = '\033[0m'    #]
    BOLD = '\033[1m'    #]
    UNDERLINE = '\033[4m' #]


class LLMChat:
    def __init__(self, host: str, prompt: str | None = None, model_name: str = 'phi-3-mini-128k-chat'):
        self.session = Session()
        self.host_url = host
        self.history = [{
            'role': 'system',
            'content': prompt 
        }]
        self.model_name = model_name
    
    def send_message(self, message: str):
        self.history.append({'role': 'user', 'content': message})
        res = self.session.post(
            urljoin(urljoin(self.host_url, 'models/'), self.model_name),
            json={
                'inputs': self.history,
                'parameters': {'use_cache': False, 'max_new_tokens': 1024}
            }
        )
        res.raise_for_status()
        res = res.json()
        self.history.append({'role': 'assistant', 'content': res[0]['generated_text']})
        return res[0]['generated_text']

    def close(self):
        self.session.close()


class TTSClient:
    def __init__(self, model_name='kokoro'):
        self.session = Session()
        self.host_url = 'http://localhost:8001'
        self.model_name = model_name

    def send(self, text, speaker='af', filename='foo.ogg'):
        res = self.session.post(
            urljoin(urljoin(self.host_url, 'models/'), self.model_name),
            json={
                'inputs': text,
                'parameters': {'speaker': speaker, 'speed': 1.2}
            }
        )
        res.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(res.content)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='llama')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--host', type=str, default='http://localhost:8001')
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLMChat(args.host, args.prompt, args.model)
    speakers = {
        'Bob': 'am_michael',
        'Laura': 'af_sky'
    }
    message = input(f'{bcolors.OKBLUE}You: ')
    print(f'{bcolors.ENDC}')
    message = f'Write a dialog between Bob and Laura. Laura thinks that {message}. Laura usually disagrees with Bob. But Bob always convinces Laura to her side. Make it sound natural and only respond with spoken language.'
    response = llm.send_message(message)
    content = response.split('\n')
    conversation = []
    for sample in content:
        if not sample or sample.startswith('[') or sample.startswith('('):  # )]
            continue
        if sample.startswith('Bob: '):
            sub = sample[len('Bob: '):]
            sub = re.sub(r'\(.*?\)', '', sub)
            sub = re.sub(r'\[.*?\]', '', sub)
            conversation.append(['Bob', sub.strip()])
        elif sample.startswith('Laura: '):
            sub = sample[len('Laura: '):]
            sub = re.sub(r'\(.*?\)', '', sub)
            sub = re.sub(r'\[.*?\]', '', sub)
            conversation.append(['Laura', sub.strip()])

    tts_client = TTSClient()

    files_out = []
    for idx, turn in enumerate(conversation):
        speaker = speakers.get(turn[0])
        text = turn[1]
        tts_client.send(text, speaker=speaker, filename=f'turn_{idx}.ogg')
        files_out.append(f'turn_{idx}.ogg')

    audio_segments = []
    for file in files_out:
        audio_segments.append(AudioSegment.from_file(file, format='flac'))
        os.remove(file)

    final_segment = AudioSegment.empty()
    for seg in audio_segments:
        final_segment = final_segment + seg

    final_segment.export('./final-out.mp3', format='mp3')


    print(f'{bcolors.OKGREEN}Bot: {response}{bcolors.ENDC}')

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        print(bcolors.ENDC, end='')
        raise e

