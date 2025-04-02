import torch
from open_omni.utils import rank0_print
from .speech_generator import SpeechGeneratorCTC


def build_speech_generator(config):
    generator_type = getattr(config, 'speech_generator_type', 'ctc')
    generator = getattr(config, 'speech_generator', None)
    if generator_type == 'ctc':
        if generator is not None:
            checkpoint = torch.load(generator, map_location="cpu")
            speech_generator = SpeechGeneratorCTC(config)
            speech_generator.load_state_dict(checkpoint, strict=False)
            rank0_print("Initialized speech generator by llama-omni")
            return speech_generator
        else:
            return SpeechGeneratorCTC(config)

    raise ValueError(f'Unknown generator type: {generator_type}')