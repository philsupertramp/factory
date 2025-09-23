from typing import Optional
from factory.utils import BaseModel
from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str
    torch_dtype: torch.dtype
    use_safetensors: bool = False
    variant: str = None
    safety_checker: str = None
    requires_safety_checker: bool = False
    vae: str = None
    scheduler: Optional[dict] = None
    feature_extractor: Optional[dict] = None


@dataclass
class ONNXModelConfig(BaseModel):
    model_id: str
    vae: str = None
    scheduler: Optional[dict] = None


@dataclass
class DiffusionPipelineConfig(BaseModel):
    base: ModelConfig
    use_scheduler: bool = False
    loras: list[str] = None
    enable_model_offload: bool = True

    _base_config_class = ModelConfig

    def __post_init__(self):
        if isinstance(self.base, dict):
            data = self.base
            if 'torch_dtype' in data:
                data['torch_dtype'] = {
                    'float16': torch.float16,
                    'bfloat16': torch.bfloat16,
                    'float32': torch.float32
                }.get(data['torch_dtype'], torch.float32)

            self.base = self._base_config_class(**data)


@dataclass
class ONNXDiffusionPipelineConfig(DiffusionPipelineConfig):
    is_sdxl: bool = False
    
    _base_config_class = ONNXModelConfig


@dataclass
class ControlNetConfig(BaseModel):
    base: str
    config: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if 'torch_dtype' in self.config:
            self.config['torch_dtype'] = (
                torch.float16
                if self.config['torch_dtype'] == 'float16'
                else torch.float32
            )

