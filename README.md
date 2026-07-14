# TabbyAPI

<p align="left">
    <img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue" alt="Python 3.10, 3.11, and 3.12">
    <a href="/LICENSE">
        <img src="https://img.shields.io/badge/License-AGPLv3-blue.svg" alt="License: AGPL v3"/>
    </a>
    <a href="https://discord.gg/sYQxnuD7Fj">
        <img src="https://img.shields.io/discord/545740643247456267.svg?logo=discord&color=blue" alt="Discord Server"/>
    </a>
</p>

<p align="left">
    <a href="https://theroyallab.github.io/tabbyAPI">
        <img src="https://img.shields.io/badge/Documentation-API-orange" alt="Developer facing API documentation">
    </a>
</p>

<p align="left">
    <a href="https://ko-fi.com/I2I3BDTSW">
        <img src="https://img.shields.io/badge/Support_on_Ko--fi-FF5E5B?logo=ko-fi&style=for-the-badge&logoColor=white" alt="Support on Ko-Fi">
    </a>
</p>

> [!IMPORTANT]
>
> In addition to the README, please read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/1.-Getting-Started) page for information about getting started!

> [!NOTE]
> 
> ExLlamaV2 models are no longer supported in the `main` branch. The last commit with ExLlamav2 
> support is preserved on the `exl2-checkpoint` branch.

> [!NOTE]
> 
> Need help? Join the [Discord Server](https://discord.gg/sYQxnuD7Fj) and get the `Tabby` role. Please be nice when asking questions.

> [!NOTE]
> 
> Tool calling support has been revamped and now no longer relies on modified Jinja templates. [See the docs for more.](docs/10.-Tool-Calling.md)

> [!NOTE]
> 
> Want to run GGUF models? Take a look at [YALS](https://github.com/theroyallab/YALS), TabbyAPI's sister project.

A FastAPI based application that allows for generating text using an LLM (large language model) using the [Exllamav3](https://github.com/turboderp-org/exllamav3) backend.

TabbyAPI is also the official API backend server for ExllamaV3.

## Disclaimer

This project is marked as rolling release. There may be bugs and changes down the line. Please be aware that you might need to reinstall dependencies if needed.

TabbyAPI is a hobby project made for a small amount of users. It is not meant to run on production servers. For that, please look at other solutions that support those workloads.

## Getting Started

> [!IMPORTANT]
> 
> Looking for more information? Check out the Wiki.

For a step-by-step guide, choose the format that works best for you:

📖 Read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/01.-Getting-Started) – Covers installation, configuration, API usage, and more.

🎥 Watch the [Video Guide](https://www.youtube.com/watch?v=03jYz0ijbUU) – A hands-on walkthrough to get you up and running quickly.

### Docker

TabbyAPI publishes a CUDA image to GitHub Container Registry. Install Docker and the NVIDIA container toolkit, then start the published image with your models directory mounted into the container:

```bash
docker pull ghcr.io/theroyallab/tabbyapi:latest
docker run --gpus all --name tabbyapi -p 5000:5000 -v /path/to/models:/app/models ghcr.io/theroyallab/tabbyapi:latest
```

Replace `/path/to/models` with the folder that contains your local model directories. The API is exposed on `http://localhost:5000`.

For Docker Compose, custom config mounts, or building the image locally, see the [Docker instructions](docs/01.-Getting-Started.md#docker).

## Features

- OpenAI compatible API
- Loading/unloading models
- HuggingFace model downloading
- Embedding model support
- JSON schema + Regex + EBNF support
- AI Horde support
- Speculative decoding via draft models
- Multi-lora with independent scaling (ex. a weight of 0.9)
- Inbuilt proxy to override client request parameters/samplers
- Flexible Jinja2 template engine for chat completions that conforms to HuggingFace
- Concurrent inference with asyncio
- Utilizes modern python paradigms
- Continuous batching engine using paged attention
- Fast classifier-free guidance
- OAI style tool/function calling

And much more. If something is missing here, PR it in!

## Supported Model Types

TabbyAPI uses Exllama as a powerful and fast backend for model inference, loading, etc. Therefore, the following types of models are supported:

- Exl3 (Highly recommended)

- FP16/BF16

In addition, TabbyAPI supports parallel batching using paged attention for Nvidia Ampere GPUs and higher.

## Contributing

Use the template when creating issues or pull requests, otherwise the developers may not look at your post.

If you have issues with the project:

- Describe the issue in detail

- If you have a feature request, please indicate it as such.

If you have a Pull Request

- Describe the pull request in detail, what, and why you are changing something

## Acknowldgements

TabbyAPI would not exist without the work of other contributors and FOSS projects:

- [ExllamaV2](https://github.com/turboderp-org/exllamav2)
- [ExllamaV3](https://github.com/turboderp-org/exllamav3)
- [Aphrodite Engine](https://github.com/PygmalionAI/Aphrodite-engine)
- [infinity-emb](https://github.com/michaelfeil/infinity)
- [FastAPI](https://github.com/fastapi/fastapi)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [SillyTavern](https://github.com/SillyTavern/SillyTavern)

## Developers and Permissions

Creators/Developers:

- [kingbri](https://github.com/bdashore3)

- [Splice86](https://github.com/Splice86)

- [Turboderp](https://github.com/turboderp)
