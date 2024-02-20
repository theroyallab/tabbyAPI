# TabbyAPI

> [!IMPORTANT]
>
>  In addition to the README, please read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/1.-Getting-Started) page for information about getting started!

> [!NOTE]
> 
>  Need help? Join the [Discord Server](https://discord.gg/sYQxnuD7Fj) and get the `Tabby` role. Please be nice when asking questions.

A FastAPI based application that allows for generating text using an LLM (large language model) using the [Exllamav2 backend](https://github.com/turboderp/exllamav2)

## Disclaimer

This API is considered as rolling release. There may be bugs and changes down the line. Please be aware that you might need to reinstall dependencies if needed.

## Getting Started

Read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/1.-Getting-Started) for more information. It contains user-facing documentation for installation, configuration, sampling, API usage, and so much more.

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
