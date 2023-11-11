
from model import ModelContainer

def progress(module, modules):
    print(f"Loaded {module}/{modules} modules")
    yield

mc = ModelContainer("/mnt/str/models/_exl2/mistral-7b-instruct-exl2/4.0bpw/", max_seq_len = 100)
mc.load(progress)

gen = mc.generate_gen("Once upon a tim", generate_window = 16, token_healing = True)
for g in gen: print(g, end = "")

mc.unload()
del mc

mc = ModelContainer("/mnt/str/models/_exl2/mistral-7b-instruct-exl2/4.65bpw/")
mc.load(progress)

response = mc.generate("All work and no play makes turbo a derpy cat.\nAll work and no play makes turbo a derpy cat.\nAll", top_k = 1)
print (response)
