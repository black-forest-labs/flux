from flux.trt.engine.base_engine import BaseEngine


class AEEngine(BaseEngine):
    def __init__(self, engine_path):
        super().__init__(engine_path)

    def call(self):
        return