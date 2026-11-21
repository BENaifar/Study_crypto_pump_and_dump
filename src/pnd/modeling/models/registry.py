"""Model registry (open/closed). [orig: run_train.py:L94-L116 model_fn dispatch]"""

from pnd.modeling.models.dnn import DNN
from pnd.modeling.models.snn import SNN
from pnd.modeling.models.snnta import SNNTA

MODELS = {"snn": SNN, "snnta": SNNTA, "dnn": DNN}


def build_model(name, tensor_dict, model_config):
    """Instantiate and ``build()`` the model registered under ``name``."""
    if name not in MODELS:
        raise ValueError("Should select correct model name:", name)
    model = MODELS[name](tensor_dict, model_config)
    model.build()
    return model
