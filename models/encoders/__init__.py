"""models/encoders - one interface, several backbones."""

from .base_encoder import BaseVisualEncoder, EncoderOutput
from .dinov3_encoder import (DINOV3_VITB16_LVD, DINOv3Config, DINOv3Encoder,
                             LoRAConfig)
from .load_encoder import (EncoderSpec, OpenCLIPGlobalEncoder, _load_encoder,
                           encoder_registry, load_encoder)

__all__ = ["BaseVisualEncoder", "EncoderOutput", "DINOv3Encoder",
           "DINOv3Config", "LoRAConfig", "DINOV3_VITB16_LVD", "EncoderSpec",
           "OpenCLIPGlobalEncoder", "load_encoder", "_load_encoder",
           "encoder_registry"]
