"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import logging
import torch
from typing import TypeAlias
from vllm import LLM, SamplingParams
from promptodile.config.qgen_config import QGenConfig

logger = logging.getLogger(__name__)


Chat: TypeAlias = dict[str, str]

class QGen:
    def __init__(self, config: QGenConfig):
        logger.info(config)
        self._config = config
        
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            logger.info('using %d CUDA GPUs', gpus)
        else:
            logger.info('no CUDA GPUs detected. Running on CPU (not recommended)')
            gpus = 1
            
        self._llm = LLM(
            model=config.model,
            tensor_parallel_size=gpus)
        
        self._system = config.prompt_templates.system
        self._user = config.prompt_templates.user
        self._assistant = config.prompt_templates.assistant
    
    @property
    def config(self) -> QGenConfig:
        return self._config
    
    @property
    def system(self) -> str | None:
        return self._system
    
    @property
    def user(self) -> str | None:
        return self._user
    
    @property
    def assistant(self) -> str | None:
        return self._assistant
    
    @staticmethod
    def format_chat(role: str, content: str) -> Chat:
        return { 'role': role, 'content': content }
    
    def get_sampling_params(self) -> SamplingParams:
        """Update and retrieve the model's sampling parameters.
        
        Uses the model's defaults where a parameter isn't specified on
        input."""
        sampling_params = self._llm.get_default_sampling_params()
        
        # Update defaults with inputs.
        for param, val in self._config.sampling_params.items():
            if hasattr(sampling_params, param):
                curr_val = getattr(sampling_params, param)
                logger.info(f'setting {param} from {curr_val} to {val}')
                setattr(sampling_params, param, val)
        
        logger.info(sampling_params)
        
        return sampling_params
    
    def chat(
        self,
        conversations: list[list[Chat]],
        sampling_params: SamplingParams,
    ) -> list[list[str]]:
        try:
            outputs = self._llm.chat( # type: ignore[attr-defined]
                conversations, # type: ignore[attr-defined]
                sampling_params,
                add_generation_prompt=True,
                use_tqdm=False)
        except ValueError as e:
            logger.exception(e)
            return [[''] * sampling_params.n] * len(conversations)
        
        output_text: list[list[str]] = []
        for output in outputs:
            output_text.append([n.text for n in output.outputs])
                
        return output_text
