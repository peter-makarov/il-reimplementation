ENCODER_MAPPING = {}
OPTIMIZER_MAPPING = {}
LR_SCHEDULER_MAPPING = {}


def register_component(name, type_):
    """
    This method can be used as a decorator to add components (encoder, optimizer or lr scheduler) to the cli interface.

    Args:
        name: name of the component
        type_: type of the component

    Returns:
        None
    """
    def register_component_cls(cls):
        if type_ == "encoder":
            if name not in ENCODER_MAPPING:
                ENCODER_MAPPING[name] = cls
        elif type_ == "optimizer":
            if name not in OPTIMIZER_MAPPING:
                OPTIMIZER_MAPPING[name] = cls
        elif type_ == "lr_scheduler":
            if name not in LR_SCHEDULER_MAPPING:
                LR_SCHEDULER_MAPPING[name] = cls

        return cls

    return register_component_cls


from .optimizers import *
from .encoders import *

