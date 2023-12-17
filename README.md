# TabbyAPI

> [!NOTE]
> 
>  Need help? Join the [Discord Server](https://discord.gg/sYQxnuD7Fj) and get the `Tabby` role. Please be nice when asking questions.

A FastAPI based application that allows for generating text using an LLM (large language model) using the https://github.com/turboderp/exllamav2.

## Disclaimer

This API is still in the alpha phase. There may be bugs and changes down the line. Please be aware that you might need to reinstall dependencies if needed.

### Help Wanted

Please check the issues page for issues that contributors can help on. We appreciate all contributions. Please read the contributions section for more details about issues and pull requests.

If you want to add samplers, add them in the [exllamav2 library](https://github.com/turboderp/exllamav2) and then link them to tabbyAPI.

## Prerequisites

To get started, make sure you have the following installed on your system:

- Python 3.x (preferably 3.11) with pip

- CUDA 12.x (you can also use CUDA 11.8 or ROCm 5.6, but there will be more work required to install dependencies such as Flash Attention 2)

NOTE: For Flash Attention 2 to work on Windows, CUDA 12.x **must** be installed!

## Installing

1. Clone this repository to your machine: `git clone https://github.com/theroyallab/tabbyAPI`

2. Navigate to the project directory: `cd tabbyAPI`

3. Create a python environment:
   
   1. Through venv (recommended)
      
      1. `python -m venv venv`
      
      2. On Windows (Using powershell or Windows terminal): `.\venv\Scripts\activate`. On Linux: `source venv/bin/activate`
   
   2. Through conda
      
      1. `conda create -n tabbyAPI python=3.11`
      
      2. `conda activate tabbyAPI`

4. Install the requirements file based on your system:
   
   1. Cuda 12.x: `pip install -r requirements.txt`
   
   2. Cuda 11.8: `pip install -r requirements-cu118.txt`
   
   3. ROCm 5.6: `pip install -r requirements-amd.txt`

## Configuration

A config.yml file is required for overriding project defaults. If you are okay with the defaults, you don't need a config file!

If you do want a config file, copy over `config_sample.yml` to `config.yml`. All the fields are commented, so make sure to read the descriptions and comment out or remove fields that you don't need.

## Launching the Application

1. Make sure you are in the project directory and entered into the venv

2. Run the tabbyAPI application: `python main.py`

## Updating

To update tabbyAPI, just run `pip install --upgrade -r requirements.txt` using the `requirements.txt` for your configuration (ex. CUDA 11.8 or ROCm 5.6)

### Update Exllamav2

> [!WARNING]
> 
> These instructions are meant for advanced users.

If the version of exllamav2 doesn't meet your specifications, you can install the dependency from various sources.

NOTE:

- TabbyAPI will print a warning if a sampler isn't found due to the exllamav2 version being too low.

- Any upgrades using a requirements file will result in overwriting your installed wheel. To fix this, change `requirements.txt` locally, create an issue or PR, or install your version of exllamav2 after upgrades.

Here are ways to install exllamav2:

1. From a [wheel/release](https://github.com/turboderp/exllamav2#method-2-install-from-release-with-prebuilt-extension) (Recommended)
   
   1. Find the version that corresponds with your cuda and python version. For example, a wheel with `cu121` and `cp311` corresponds to CUDA 12.1 and python 3.11

2. From [pip](https://github.com/turboderp/exllamav2#method-3-install-from-pypi): `pip install exllamav2`
   
   1. This is a JIT compiled extension, which means that the initial launch of tabbyAPI will take some time. The build may also not work due to improper environment configuration.

3. From [source](https://github.com/turboderp/exllamav2#method-1-install-from-source)

## API Documentation

Docs can be accessed once you launch the API at `http://<your-IP>:<your-port>/docs`

If you use the default YAML config, it's accessible at `http://localhost:5000/docs` 

## Authentication

TabbyAPI uses an API key and admin key to authenticate a user's request. On first launch of the API, a file called `api_tokens.yml` will be generated with fields for the admin and API keys.

If you feel that the keys have been compromised, delete `api_tokens.yml` and the API will generate new keys for you.

API keys and admin keys can be provided via the following request headers:

- `x-api-key` and `x-admin-key` respectively

- `Authorization` with the `Bearer ` prefix

DO NOT share your admin key unless you want someone else to load/unload a model from your system!

#### Authentication Requrirements

All routes require an API key except for the following which require an **admin** key

- `/v1/model/load`

- `/v1/model/unload`

## Chat Completions

`/v1/chat/completions` now uses Jinja2 for templating. Please read [Huggingface's documentation](https://huggingface.co/docs/transformers/main/chat_templating) for more information of how chat templates work.

Also make sure to set the template name in `config.yml` to the template's filename.

## Common Issues

- AMD cards will error out with flash attention installed, even if the config option is set to False. Run `pip uninstall flash_attn` to remove the wheel from your system.
  
  - See [#5](https://github.com/theroyallab/tabbyAPI/issues/5)

- Exllamav2 may error with the following exception: `ImportError: DLL load failed while importing exllamav2_ext: The specified module could not be found.`
  
  - First, make sure to check if the wheel is equivalent to your python version and CUDA version. Also make sure you're in a venv or conda environment.
  
  - If those prerequisites are correct, the torch cache may need to be cleared. This is due to a mismatching exllamav2_ext.
    
    - In Windows: Find the cache at `C:\Users\<User>\AppData\Local\torch_extensions\torch_extensions\Cache` where `<User>` is your Windows username
    
    - In Linux: Find the cache at `~/.cache/torch_extensions`
    
    - look for any folder named `exllamav2_ext` in the python subdirectories and delete them.
    
    - Restart TabbyAPI and launching should work again.

## Supported Model Types

TabbyAPI uses Exllamav2 as a powerful and fast backend for model inference, loading, etc. Therefore, the following types of models are supported:

- Exl2 (Highly recommended)

- GPTQ

- FP16 (using Exllamav2's loader)

#### Alternative Loaders/Backends

If you want to use a different model type than the ones listed above, here are some alternative backends with their own APIs:

- GGUF + GGML - [KoboldCPP](https://github.com/lostruins/KoboldCPP)

- AWQ - [Aphrodite Engine](https://github.com/PygmalionAI/Aphrodite-engine)

- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)

## Contributing

If you have issues with the project:

- Describe the issues in detail

- If you have a feature request, please indicate it as such.

If you have a Pull Request

- Describe the pull request in detail, what, and why you are changing something

## Developers and Permissions

Creators/Developers:

- [kingbri](https://github.com/bdashore3)

- [Splice86](https://github.com/Splice86)

- [Turboderp](https://github.com/turboderp)
