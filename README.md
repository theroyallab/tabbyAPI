# TabbyAPI

A FastAPI based application that allows for generating text using an LLM (large language model) using the [exllamav2 backend](https://github.com/turboderp/exllamav2).

## Disclaimer

This API is still in the alpha phase. There may be bugs and changes down the line. Please be aware that you might need to reinstall dependencies if needed.

## Prerequisites

To get started, make sure you have the following installed on your system:

- Python 3.x (preferably 3.11) with pip

- CUDA 12.1 or 11.8

NOTE: For Flash Attention 2 to work on Windows, CUDA 12.1 **must** be installed!

## Installing

1. Clone this repository to your machine: `git clone https://github.com/theroyallab/tabbyAPI`

2. Navigate to the project directory: `cd tabbyAPI`

3. Create a virtual environment:
   
   1. `python -m venv venv`
   
   2. On Windows: `.\venv\Scripts\activate`. On Linux: `source venv/bin/activate`

4. Install torch using the instructions found [here](https://pytorch.org/get-started/locally/)

5. Install an exllamav2 wheel from [here](https://github.com/turboderp/exllamav2/releases):
   
   1. Find the version that corresponds with your cuda and python version. For example, a wheel with `cu121` and `cp311` corresponds to CUDA 12.1 and python 3.11

6. Install the other requirements via: `pip install -r requirements.txt`

## Configuration

Copy over `config_sample.yml` to `config.yml`. All the fields are commented, so make sure to read the descriptions and comment out or remove fields that you don't need.

## Launching the Application

1. Make sure you are in the project directory and entered into the venv

2. Run the tabbyAPI application: `python main.py`

## API Documentation

Docs can be accessed once you launch the API at `http://<your-IP>:<your-port>/docs`

If you use the default YAML config, it's accessible at `http://localhost:5000/docs` 

## Authentication

TabbyAPI uses an API key and admin key to authenticate a user's request. On first launch of the API, a file called `api_tokens.yml` will be generated with fields for the admin and API keys.

If you feel that the keys have been compromised, delete `api_tokens.yml` and the API will generate new keys for you.

API keys and admin keys can be provided via:

- `x-api-key` and `x-admin-key` respectively

- `Authorization` with the `Bearer ` prefix

DO NOT share your admin key unless you want someone else to load/unload a model from your system!

#### Authentication Requrirements

All routes require an API key except for the following which require an **admin** key

- `/v1/model/load`

- `/v1/model/unload`

## Contributing

If you have issues with the project:

- Describe the issues in detail

- If you have a feature request, please indicate it as such.

If you have a Pull Request

- Describe the pull request in detail, what, and why you are changing something

## Developers and Permissions

Creators/Developers:

- kingbri

- Splice86

- Turboderp
