""" Test the model container. """
from model import ModelContainer


def progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


def test_load_gen(model_path):
    """Test loading a model."""
    container = ModelContainer(model_path)
    loader = container.load_gen(progress)
    for module, modules in loader:
        print(module, modules)
    container.unload()
    del container


def test_generate_gen(model_path):
    """Test generating from a model."""
    container = ModelContainer(model_path)
    generator = container.generate_gen("Once upon a tim", token_healing=True)
    for chunk in generator:
        print(chunk, end="")
    container.unload()
    del container


def test_generate(model_path):
    """Test generating from a model."""
    model_container = ModelContainer(model_path)
    model_container.load(progress)
    prompt = (
        "All work and no play makes turbo a derpy cat.\n"
        "All work and no play makes turbo a derpy cat.\nAll"
    )
    response = model_container.generate(
        prompt, top_k=1, max_new_tokens=1000, stream_interval=0.5
    )
    print(response)


if __name__ == "__main__":
    MODEL1 = "/mnt/str/models/_exl2/mistral-7b-instruct-exl2/4.0bpw/"
    MODEL2 = "/mnt/str/models/_exl2/mistral-7b-instruct-exl2/4.65bpw/"
    test_load_gen(MODEL1)
    test_generate_gen(MODEL1)
    test_generate(MODEL2)
