from factory.ml.pipelines.general import PipelineMixin
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import onnxruntime_genai as og
import time


class SummarizationPipeline(PipelineMixin):
    output_type = 'text'
    
    def __init__(self, model_name='pszemraj/led-large-book-summary'):
        self.model_name = model_name
        self.device = 'cpu' #cuda' if torch.cuda.is_available() else 'cpu'

    def _load_pipeline(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-text"

    def run_task(self, task):
        if task.task == "text-to-text":
            return self.text_to_text(task.inputs, task.parameters)
        raise ValueError("Invalid task")

    def get_options(self):
        return {
            'task': 'text-to-text',
            'output_type': 'text',
            'parameters': {
                'inputs': 'A text to summarize',
                'parameters': {
                    'batch_length': 'Number of tokens to summarise as a batch.',
                    'batch_stride': 'Number of tokens to overlap between each batches.'
                }
            }
        }

    def summarize_and_score(self, ids, mask, **kwargs):
        ids = ids[None, :]
        mask = mask[None, :]
        
        input_ids = ids.to(self.device)
        attention_mask = mask.to(self.device)
        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1

        summary_pred_ids = self.model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
        summary = self.tokenizer.batch_decode(
            summary_pred_ids.sequences, 
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
        score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)
        
        return summary, score
        
    def summarize_via_tokenbatches(
            self,
            input_text:str,
            batch_length=8192,
            batch_stride=16,
            **kwargs,
        ):
        
        encoded_input = self.tokenizer(
            input_text, 
            padding='max_length', 
            truncation=True,
            max_length=batch_length, 
            stride=batch_stride,
            return_overflowing_tokens=True,
            add_special_tokens =False,
            return_tensors='pt',
        )
        
        in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask
        gen_summaries = []
        for _id, _mask in zip(in_id_arr, att_arr):

            result, score = self.summarize_and_score(
                ids=_id, 
                mask=_mask, 
                #**kwargs,
            )
            score = round(float(score),4)
            _sum = {
                "input_tokens":_id,
                "summary":result,
                "summary_score":score,
            }
            gen_summaries.append(_sum)
            print(f"\t{result[0]}\nScore:\t{score}")

        return gen_summaries

    def text_to_text(self, text, options):
        return self.summarize_via_tokenbatches(text, **options)


class ChatPipelineMixin(PipelineMixin):
    output_type = 'text'
    
    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "chat-completion"

    def run_task(self, task):
        if task.task == "chat-completion":
            return self.chat_completion(task.inputs, task.parameters)
        raise ValueError("Invalid task")

    def get_options(self):
        return {
            'task': 'text-to-text',
            'output_type': 'text',
            'parameters': {
                'inputs': 'A text to chat about',
                'temperature': 'The temperature of the model',
                'max_new_tokens': 'The maximum number of tokens to generate',
                'top_k': 'The top k tokens to sample from',
                'top_p': 'The top p tokens to sample from',
                'repetition_penalty': 'The repetition penalty',
                'num_beams': 'The number of beams to use when searching for the next token',
            }
        }


class ChatPipeline(ChatPipelineMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()

    def _load_pipeline(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )#.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def chat_completion(self, messages, options):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            **options,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return [{"generated_text": response}]


class ONNXChatPipeline(ChatPipelineMixin):
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        self.model_name = model_name
        self.chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
        self.history_template = '<|user|>\n{input} <|end|>\n<|assistant|>\n{response} <|end|>'
        self.allowed_options = [
            'do_sample', 'max_length', 'min_length', 
            'top_p', 'top_k', 'temperature', 'repetition_penalty'
        ]

    def _load_pipeline(self):
        self.model = og.Model(self.model_name)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()

    def chat_completion(self, messages, options, timings=True, verbose=True):
        search_options = {name:options.get(name) for name in self.allowed_options if name in options}

        history = []
        for message in messages:
            if message['role'] == 'user':
                history.append(f'<|user|>\n{message["content"]} <|end|>')
            elif message['role'] == 'assistant':
                history.append(f'<|assistant|>\n{message["content"]} <|end|>')
            elif message['role'] == 'system':
                history.append(f'<|system|>\n{message["content"]} <|end|>')
            else:
                print(f"Error: Unknown role {message['role']}")

        clean_chat_history = list(filter(lambda x: x['role'] not in ['system'], messages))
        if len(clean_chat_history) % 2 == 0 and len(clean_chat_history) > 0:
            raise ValueError("Error: The history should have an even number of messages. The last message should be from the user.")
        

        prompt = '\n'.join(history)

        # Set the max length to something sensible by default, unless it is specified by the user,
        # since otherwise it will be set to the entire context length
        if 'max_length' not in search_options:
            search_options['max_length'] = 2048


        # Keep asking for input prompts in a loop
        if not prompt:
            print("Error, input cannot be empty")
            return

        prompt += '\n<|assistant|>\n'
        if timings: started_timestamp = time.time()

        # If there is a chat template, use it
        input_tokens = self.tokenizer.encode(prompt)

        params = og.GeneratorParams(self.model)
        params.try_use_cuda_graph_with_max_batch_size(1)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(self.model, params)
        new_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            new_tokens.append(self.tokenizer_stream.decode(new_token))

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

        return [{"generated_text": ''.join(new_tokens)}]
