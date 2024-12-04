from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.modules import PyModule


class Entity:
    pass


class ModuleEntity(Entity):
    def __init__(self, module: "PyModule"):
        self.module = module
