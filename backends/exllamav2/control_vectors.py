"""Control vector support for ExLlamaV2 models."""

import glob
import torch
from gguf.gguf_reader import GGUFReader

class ExLlamaV2ModuleWrapper:
    @classmethod
    def wrap(cls, model, vector_configs):
        vectors = {}
        for file in glob.glob(str(model.config.model_dir) + '-vectors/*.gguf'):
            base = file.rsplit('-', 1)[-1].replace('.gguf', '')
            vector, direction = base.split('__')
            print(f"Loaded control vector: {vector}, Direction: {direction}")
            reader = GGUFReader(file)
            if reader.tensors[0].n_elements != model.config.hidden_size:
                print(f' ## Control vector n_elements ({reader.tensors[0].n_elements}) != model.config.hidden_size ({model.config.hidden_size})')
                return
            layers = torch.zeros((model.config.num_hidden_layers, model.config.hidden_size), dtype=torch.float32)
            for tensor in reader.tensors:
                idx = int(tensor.name.split('.')[-1])
                layers[idx] = torch.from_numpy(tensor.data.copy())
            vectors.setdefault(vector, {})[direction] = layers
        
        vector_configs = vector_configs.split(',')
        control_vector = torch.zeros((model.config.num_hidden_layers, model.config.hidden_size), dtype=torch.float32)
        for vector_config in vector_configs:
            (vector, direction, weight) = vector_config.split(':')
            vector_dirs = None
            for k, v in vectors.items():
                if vector in k:
                    vector = k
                    vector_dirs = v
                    break
            if vector_dirs is None:
                print(f' !! Error: No vector for "{vector}" ({vector_config})')
                continue
            debias_layers = vector_dirs.get('debias', None)
            if debias_layers is None:
                print(f' !! Error: No debias for "{vector}" ({vector_config})')
                continue
            direction_layers = vector_dirs.get(direction, None)
            if direction_layers is None:
                print(f' !! Error: No "{direction}" for "{vector}" ({vector_config})')
                continue
            try:
                weight = float(weight)
            except Exception as e:
                print(f' !! Non float weight {weight} ({vector_config})')
                weight = 1.0
            print(f' -- Applying {vector} debias and {direction} * {weight}')
            control_vector += debias_layers
            control_vector += direction_layers * weight

        for idx, module in enumerate(model.modules):
            if idx == 0 or idx >= (len(model.modules) - 2) or module.name != 'MLP':
                continue
            model.modules[idx] = ExLlamaV2ModuleWrapper(module, control_vector)

    def __init__(self, module, control_vector):
        self.module = module 
        self.control_vector = control_vector

    def __getattribute__(self, name):
        if name == 'forward':
            return object.__getattribute__(self, 'wrapped_forward')
        try:
            return getattr(object.__getattribute__(self, 'module'), name)
        except AttributeError:
            pass
        return object.__getattribute__(self, name)

    def wrapped_forward(self, *args, **kwargs):
        x = self.module.forward(*args, **kwargs)
        try:
            prev_norm = torch.norm(x, p=2)
            x += self.control_vector[self.module.layer_idx].clone().to(x.device)
            x *= prev_norm / torch.norm(x, p=2)
        except IndexError:
            pass
        return x
