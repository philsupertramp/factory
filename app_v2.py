import base64
from typing import Union, Annotated
import cv2
import enum
import json
import numpy as np
import torch
import traceback

from factory import config

from fastapi import FastAPI, Response, File, UploadFile, Form
from pydantic import BaseModel
from io import BytesIO
from factory.ml import manager
from factory.api.models import (
    GenerationRequest, TextToImageRequest, ImageToImageRequest,
    TextToSpeechRequest, TextGenerationRequest, ChatCompletionRequest,
    SpeechToTextRequest
)
from tempfile import NamedTemporaryFile
from PIL import Image


app = FastAPI()


def send_image_response_base64(results: bytes, format: str = "image/jpeg") -> Response:
    response = []
    if format == 'image/jpeg':
        for result in results:
            buffer = BytesIO()
            result.save(buffer, format=format.split('/')[-1])
            print('converted to buffer')
            response.append(buffer.getvalue())
    elif 'audio' in format:
        response = [results]
    else:
        if isinstance(results, list):
            if isinstance(results[0], dict):
                for result in results:
                    response.append({
                        key: (
                            value.tolist()
                            if isinstance(value, (np.ndarray, torch.Tensor))
                            else value
                        )
                        for key, value in result.items()
                    })
                response = json.dumps(response)
            else:
                response = json.dumps(results)
        elif isinstance(results, dict):
            response = json.dumps(results)
        else:
            response = json.dumps([results.tolist()])
    return Response(
        content=response[0] if len(response) == 1 else response,
        media_type=format
    )


response_type_map = {
    'image': 'image/jpeg',
    'audio': 'audio/flac',
    'text': 'application/json',
}


@app.post('/models/{model_name}')
async def generate(
        model_name: str,
        q: GenerationRequest | dict | None | bytes | str = None,
        response: Response = None
    ):
    if not config.LOAD_MODELS:
        response.status_code = 503
        return {'error': 'Models are not loaded'}

    is_multi = 'multi' in model_name
    model_name = model_name.replace('-multi', '').replace('_', '/', 1)
    model = manager.get_model(model_name)
    if model is None:
        response.status_code = 400
        print('Model not found')
        return {'error': 'Model not found'}

    task = model.get_task(is_multi)
    task_params = {}
    if isinstance(q, dict):
        task_params = {**q}
    elif isinstance(q, bytes):
        task_params['inputs'] = q
        print('Setting inputs from bytes')

    task = {
        'text-to-image': TextToImageRequest,
        'image-to-image': ImageToImageRequest,
        'image-to-image-multi': ImageToImageRequest,
        'text-to-speech': TextToSpeechRequest,
        'text-to-text': TextGenerationRequest,
        'chat-completion': ChatCompletionRequest,
        'automatic-speech-recognition': SpeechToTextRequest,
    }.get(task)(**{**task_params, 'task': task, 'is_multi': is_multi})
    try:
        results = model.run_task(task)
    except ValueError as e:
        response.status_code = 400
        print(f'Error for {q} with task {task} using {task_params}: {e}')
        tb = traceback.format_exc()
        print(tb)
        return {'error': str(e)}
    return send_image_response_base64(results, response_type_map.get(model.output_type))


@app.post('/models/{model_name}/v1/chat/completions')
async def chat_completions(model_name: str, q: ChatCompletionRequest, response: Response):
    if not config.LOAD_MODELS:
        response.status_code = 503
        return {'error': 'Models are not loaded'}

    return await generate(model_name, q.dict(), response)


@app.get('/models')
async def get_models():
    return {
        model_name: model.get_options()
        for model_name, model in manager.get_all_models().items()
    }


@app.get('/models/{model_name}')
async def get_model(model_name: str):
    model = manager.get_model(model_name)
    if model is None:
        return {'error': 'Model not found'}
    return model.get_options()
