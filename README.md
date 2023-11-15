
# tabbyAPI

tabbyAPI is a FastAPI-based application that provides an API for generating text using a language model. This README provides instructions on how to launch and use the tabbyAPI.

## Prerequisites

Before you get started, ensure you have the following prerequisites installed on your system:

- Python 3.x (with pip)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository to your local machine:

git clone https://github.com/Splice86/tabbyAPI.git


2. Navigate to the project directory:

cd tabbyAPI


3. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate


4. Install project dependencies using pip:

pip install -r requirements.txt


5. Install exllamav2 to your venv

git clone https://github.com/turboderp/exllamav2.git

cd exllamav2

pip install -r requirements.txt

python setup.py install



## Launch the tabbyAPI Application

To start the tabbyAPI application, follow these steps:

1. Ensure you are in the project directory and the virtual environment is activated (if used).

2. Run the tabbyAPI application:


python main.py

3. The tabbyAPI application should now be running. You can access it by opening a web browser and navigating to `http://localhost:8000` (if running locally).

## Usage

The tabbyAPI application provides the following endpoint:

- '/v1/model' Retrieves information about the currently loaded model.
- '/v1/model/load' Loads a new model based on provided data and model configuration.
- '/v1/model/unload' Unloads the currently loaded model from the system.
- '/v1/completions' Use this endpoint to generate text based on the provided input data.

### Example Request (using `curl`)

curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 2261702e8a220c6c4671a264cd1236ce" \
  -d '{
    "model": "airoboros-mistral2.2-7b-exl2",
    "prompt": ["A tabby","is"],
    "stream": true,
    "top_p": 0.73,
    "stop": "[",
    "max_tokens": 360,
    "temperature": 0.8,
    "mirostat_mode": 2,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1
  }' \
  http://127.0.0.1:8012/v1/completions



### Parameter Guide

*note* This stuff still needs to be expanded and updated

{
  "model": "airoboros-mistral2.2-7b-exl2",
  "prompt": ["A tabby","is"],
  "stream": true,
  "top_p": 0.73,
  "stop": "[",
  "max_tokens": 360,
  "temperature": 0.8,
  "mirostat_mode": 2,
  "mirostat_tau": 5,
  "mirostat_eta": 0.1
}

Model: "airoboros-mistral2.2-7b-exl2"
    This specifies the specific language model being used. It's essential for the API to know which model to employ for generating responses.

Prompt: ["Hello there! My name is", "Brian", "and I am", "an AI"]
    The prompt *QUESTION* why is it a list of strings instead of a single string? 
Stream: true
    Whether the response should be streamed back or not.

Top_p: 0.73
    cumulative probability threshold

Stop: "["
    The stop parameter defines a string that stops the generation.

Max_tokens: 360
    This parameter determines the maximum number of tokens.

Temperature: 0.8
    Temperature controls the randomness of the generated text.

Mirostat_mode: 2
   ?
Mirostat_tau: 5
   ?
Mirostat_eta: 0.1
   ?
