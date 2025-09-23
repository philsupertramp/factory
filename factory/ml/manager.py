from .pipelines import (
    IPFaceIDPipeline, IPPipeline,
    DiffusionModel, ONNXDiffusionModel,
    Speech5TTSPipeline, BarkTTSPipeline,
    SummarizationPipeline, QRCodePipeline,
    ChatPipeline, SpeechToTextPipeline,
    ONNXChatPipeline, HARTPipeline,
    SanaPipeline, KokoroTTSPipeline
)
import yaml


available_constructors = {
    # Diffuser and Transformer models
    IPPipeline.__name__: IPPipeline,
    IPFaceIDPipeline.__name__: IPFaceIDPipeline,
    DiffusionModel.__name__: DiffusionModel,
    Speech5TTSPipeline.__name__: Speech5TTSPipeline,
    BarkTTSPipeline.__name__: BarkTTSPipeline,
    SummarizationPipeline.__name__: SummarizationPipeline,
    QRCodePipeline.__name__: QRCodePipeline,
    ChatPipeline.__name__: ChatPipeline,
    SpeechToTextPipeline.__name__: SpeechToTextPipeline,
    KokoroTTSPipeline.__name__: KokoroTTSPipeline,
    # ONNX support
    ONNXDiffusionModel.__name__: ONNXDiffusionModel,
    ONNXChatPipeline.__name__: ONNXChatPipeline,

    # HART
    HARTPipeline.__name__: HARTPipeline,
    SanaPipeline.__name__: SanaPipeline,
}


class ModelManager:
    def __init__(self, config_file='./models/model_manager.yaml', load_models=True):
        fstream = open(config_file, 'r')
        self.config = yaml.load(fstream, Loader=yaml.FullLoader)
        fstream.close()
        self.models = {'base': {}, 'models': {}}
        self.require_model_load = load_models
        self.load_models()

    def load_models(self):
        # load base models
        for model_name, config in self.config.get('base_models', {}).items():
            model = self.load_model(config)
            if self.require_model_load:
                model._load_pipeline()
            self.models['base'][model_name] = model

        # load ensemble models
        for model_name, config in self.config.get('models', {}).items():
            model = self.load_model(config)
            if self.require_model_load:
                model._load_pipeline()
            self.models['models'][model_name] = model

    def load_model(self, config):
        constructor = config.get('constructor')
        args = config.get('args', [])
        kwargs = config.get('kwargs', {})
        base_model = config.get('base_model', None)

        if constructor not in available_constructors:
            raise ValueError(f"Constructor {constructor} not found in available_constructors")

        if base_model:
            return available_constructors[constructor](self.models['base'][base_model].base, *args, **kwargs)

        return available_constructors[constructor](*args, **kwargs)

    def get_model(self, model_name):
        is_base_model = model_name in self.models['base']
        return self.models['base' if is_base_model else 'models'].get(model_name)

    def get_all_models(self):
        return {**self.models['base'], **self.models['models']}

