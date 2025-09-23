import os
import json
import random
import torch
import torch._dynamo
import numpy as np
import onnxruntime_genai as og
from PIL import Image

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from optimum.onnxruntime import ORTStableDiffusionPipeline, ORTStableDiffusionXLPipeline

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig,
)

from factory.ml.models import DiffusionPipelineConfig, ONNXDiffusionPipelineConfig
from factory.ml.pipelines.general import PipelineMixin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionModel(PipelineMixin):
    output_type = "image"

    def __init__(self, model_name):
        model_config = {}
        # parse model configs from directory
        for file in os.listdir('./models/configs'):
            if file.endswith(".json") and model_name.replace('/', '_') in file and not 'onnx' in file:
                with open(os.path.join('./models/configs/', file)) as f:
                    model_config = DiffusionPipelineConfig(**json.load(f))

        self.model_config = model_config

        self.model_params = {
            "num_inference_steps": 1,
        }

    def _load_pipeline(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = self.model_config.base.dict()
        vae = model_args.pop('vae', None)
        scheduler = model_args.pop('scheduler', None)
        if scheduler:
            model_args['scheduler'] = DDIMScheduler(**scheduler['args'])
        if vae:
            model_args['vae'] = AutoencoderKL.from_pretrained(vae).to(dtype=torch.float16)

        self.base = DiffusionPipeline.from_pretrained(**model_args).to(device)
        
        
        if self.model_config.use_scheduler:
            config = self.base.scheduler.config
            extra_config = {}
            if "algorithm_type" in config and config.get("algorithm_type") == "deis":
                extra_config["algorithm_type"] = "sigma_min"
            config = dict(**{**config, **extra_config})
            self.base.scheduler = DPMSolverMultistepScheduler.from_config(config)

        if self.model_config.loras:
            self.base.load_lora_weights(self.model_config.loras[0])

        self.base.unet.set_default_attn_processor()
        self.base.vae.set_default_attn_processor()

        if self.model_config.enable_model_offload:
            torch.cuda.empty_cache()
            self.base.enable_sequential_cpu_offload()

    def get_options(self):
        return {
            'task': 'text-to-image',
            'output_type': self.output_type,
            'parameters': {
                'inputs': 'An image of a cat',
                **self.model_params,
            }
        }

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-image"

    def run_task(self, task):
        if task.task == "text-to-image":
            return self.text_to_image(task.inputs, task.parameters.dict() or self.model_params)
        raise ValueError("Invalid task")

    def build_embeddings(self, enhanced_prompt, negative_prompt=None):
        max_length = self.base.tokenizer.model_max_length

        input_ids = self.base.tokenizer(enhanced_prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")

        negative_ids = self.base.tokenizer(
            negative_prompt or "",
            truncation=False,
            padding="max_length",
            max_length=input_ids.shape[-1],
            return_tensors="pt"
        ).input_ids
        negative_ids = negative_ids.to("cuda")

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(self.base.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(self.base.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
        return prompt_embeds, negative_prompt_embeds

    def text_to_image(self, prompt, options):
        negative_prompt = options.pop('negative_prompt', None)
        prompt_embeds, pooled_embeds = self.base.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            device='cpu',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )
        if isinstance(prompt_embeds, (list, tuple)):
            negative_prompt_embeds = prompt_embeds[1]
            prompt_embeds = prompt_embeds[0]
            negative_pooled_prompt_embeds = pooled_embeds[1]
            pooled_prompt_embeds = pooled_embeds[0]
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            pooled_prompt_embeds = pooled_embeds

        return self.base(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            **{**self.model_params, **options}
        ).images


class ONNXDiffusionModel(DiffusionModel):
    def __init__(self, model_name):
        model_config = {}
        # parse model configs from directory
        for file in os.listdir('./models/configs'):
            if file.endswith(".json") and model_name.replace('/', '_') in file and 'onnx' in file:
                with open(os.path.join('./models/configs/', file)) as f:
                    model_config = ONNXDiffusionPipelineConfig(**json.load(f))

        self.model_config = model_config

        self.model_params = {
            "num_inference_steps": 1,
        }

    def _load_pipeline(self):
        model_args = self.model_config.base.dict()
        vae = model_args.pop('vae', None)
        scheduler = model_args.pop('scheduler', None)
        if scheduler:
            model_args['scheduler'] = DDIMScheduler(**scheduler['args'])
        if vae:
            model_args['vae'] = AutoencoderKL.from_pretrained(vae).to(dtype=torch.float16)

        if self.model_config.is_sdxl:
            pipeline = ORTStableDiffusionPipeline
        else:
            pipeline = ORTStableDiffusionXLPipeline
            
        self.base = pipeline.from_pretrained(**model_args)#, export=True)#.to(device)
        self.allowed_params = [
            'prompt', 'height', 'width', 'num_inference_steps', 'guidance_scale', 'negative_prompt', 'num_images_per_prompt',
            'eta' 'generator', 'latents', 'prompt_embeds', 'negative_prompt_embeds', 'output_type', 'return_dict', 'callback'
            'callback_steps', 'guidance_rescale',
        ]

    def get_options(self):
        return {
            'task': 'text-to-image',
            'output_type': self.output_type,
            'parameters': {
                'inputs': 'An image of a cat',
                **{**self.model_params, **{k: '' for k in self.allowed_params}},
            }
        }

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-image"

    def run_task(self, task):
        if task.task == "text-to-image":
            return self.text_to_image(task.inputs, task.parameters.dict() or self.model_params)
        raise ValueError("Invalid task")
    
    def text_to_image(self, prompt, options):
        negative_prompt = options.pop('negative_prompt', None)
        options = {
            k: v
            for k, v in {**self.model_params, **options}.items()
            if k in self.allowed_params
        }
        return self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **options
        ).images


class HARTPipeline(PipelineMixin):
    output_type = "image"
    
    def __init__(self, max_token_length=300, use_ema=False, device='cuda'):
        self.model_params = {
            "seed": 0,
            "guidance_scale": 4.5,
            "randomize_seed": False,
            "more_smooth": True,
            "enhance_prompt": True
        }
        self.max_token_length = max_token_length
        self.use_ema = use_ema
        self.device = device

    def _load_pipeline(self):
        model_path = './models/HART/hart-0.7b-1024px/llm'
        text_model_path = './models/HART/Qwen2-VL-1.5B-Instruct'
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.use_ema:
            self.model.load_state_dict(
                torch.load(os.path.join(model_path, "ema_model.bin"))
            )

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModel.from_pretrained(text_model_path, quantization_config=quantization_config)#.to('cpu')
        self.text_model.eval()
        self.text_tokenizer_max_length = self.max_token_length

    def randomize_seed_fn(self, seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, 999999)
        return seed

    def generate(
        self,
        prompt: str,
        seed: int = 0,
        # width: int = 1024,
        # height: int = 1024,
        guidance_scale: float = 4.5,
        randomize_seed: bool = True,
        more_smooth: bool = True,
        enhance_prompt: bool = True,
    ):
        from hart.utils import encode_prompts, llm_system_prompt
        # pipe.to(device)
        seed = int(self.randomize_seed_fn(seed, randomize_seed))
        generator = torch.Generator().manual_seed(seed)

        prompts = [prompt]

        with torch.inference_mode():
            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                prompts,
                self.text_model,
                self.text_tokenizer,
                self.max_token_length,
                llm_system_prompt,
                enhance_prompt,
            )

            infer_func = self.model.autoregressive_infer_cfg
            device = self.device
            with torch.autocast(
                device, enabled=True, dtype=torch.float16, cache_enabled=True
            ):

                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor.to(device),
                    cfg=guidance_scale,
                    g_seed=seed,
                    more_smooth=more_smooth,#args.more_smooth,
                    context_position_ids=context_position_ids.to(device),
                    context_mask=context_mask.to(device),
                    num_maskgit_iters=1
                ).float()

        # bs, 3, r, r
        images = []
        sample_imgs_np = output_imgs.clone().mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            images.append(cur_img_store)

        return images

    def run_task(self, task):
        params = task.parameters.dict()
        filtered_params = {k: v for k, v in params.items() if k in self.model_params}
        generation = self.generate(task.inputs, **filtered_params)
        # clear caches
        torch.cuda.empty_cache()
        return generation

    def get_options(self):
        return {
            'task': 'text-to-image',
            'output_type': self.output_type,
            'parameters': {
                'inputs': 'An image of a cat',
                **self.model_params,
            }
        }

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-image"


class SanaPipeline(PipelineMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_params = {
            "num_inference_steps": 1,
            'negative_prompt': '',
            'num_images_per_prompt': 1,
            'guidance_scale': 4.5,
            'eta': 0.0,
            'height': 512,
            'width': 512,
            'complex_human_instruction': ['instructions']
        }
        self.device = torch.device('cpu')#"cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

    def _load_pipeline(self):
        from diffusers import SanaPipeline as HFSanaPipeline
        self.pipe = HFSanaPipeline.from_pretrained(self.model_name, variant='fp16')
        self.pipe.to(self.device)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.transformer = self.pipe.transformer.to(torch.float16)

    def get_options(self):
        return {
            'task': 'text-to-image',
            'output_type': 'image',
            'parameters': {
                'inputs': 'An image of a cat',
                **self.model_params,
            }
        }

    def get_task(self, is_multi):
        if is_multi:
            raise ValueError("Invalid task")
        return "text-to-image"

    def run_task(self, task):
        params = task.parameters.dict()
        filtered_params = {k: v for k, v in params.items() if k in self.model_params}
        generation = self.pipe(task.inputs, **filtered_params)[0]
        # clear caches
        torch.cuda.empty_cache()
        return generation

