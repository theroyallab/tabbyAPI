
from model import ModelContainer

def progress(module, modules):
    yield module, modules

container = ModelContainer("/mnt/str/models/_exl2/mistral-7b-instruct-exl2/4.0bpw/")
loader = container.load_gen(progress)
for (module, modules) in loader:
    print(module, modules)

generator = container.generate_gen("Once upon a tim", token_healing = True)
for g in generator: print(g, end = "")

container.unload()
del container

mc = ModelContainer("/mnt/str/models/_exl2/mistral-7b-instruct-exl2/4.65bpw/")
mc.load(progress)

response = mc.generate("All work and no play makes turbo a derpy cat.\nAll work and no play makes turbo a derpy cat.\nAll", top_k = 1, max_new_tokens = 1000, stream_interval = 0.5)
print (response)
