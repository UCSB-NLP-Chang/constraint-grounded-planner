from .simenv.box1 import Box1Env
from .simenv.box3d.box3d import Box3DEnv

from inference.registry_utils import Registry

ENV_REGISTRY = Registry("environment")
ENV_REGISTRY.register("box1", Box1Env)
ENV_REGISTRY.register("box3d", Box3DEnv)
