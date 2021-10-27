<<<<<<< Updated upstream
<<<<<<< Updated upstream
from .generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3, contrastive_gpt2
from .gpt2_generation import GPT2Generation
from .dexperts_generation import DExpertsGeneration
from .dexperts_gpt3_generation import DExpertsGPT3Generation
from .pplm_generation import PPLMGeneration
from .contrastive_generation import ContrastiveGeneration

__all__ = [
    'gpt2',
    'gpt3',
    'pplm',
    'dexperts',
    'dexperts_gpt3',
    'contrastive_gpt2'
]
=======
from .generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3, cont
=======
from .generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3, cont, mapping, att, attadd
>>>>>>> Stashed changes
from .gpt2_generation import GPT2Generation
from .dexperts_generation import DExpertsGeneration
from .dexperts_gpt3_generation import DExpertsGPT3Generation
from .pplm_generation import PPLMGeneration
from .cont_generation import ContGeneration
<<<<<<< Updated upstream
=======
from .mapping_generation import MappingGeneration
from .att_generation import AttGeneration
from .att_add_generation import AttAddGeneration
>>>>>>> Stashed changes
__all__ = [
    'gpt2',
    'gpt3',
    'pplm',
    'dexperts',
    'dexperts_gpt3',
<<<<<<< Updated upstream
    'cont'
]
>>>>>>> Stashed changes
=======
    'cont',
    'mapping',
    'att',
    'attadd'
]
>>>>>>> Stashed changes
