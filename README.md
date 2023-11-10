
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

2. Run the tabbyAPI application using Uvicorn:


uvicorn main:app --host 0.0.0.0 --port 8000 --reload


- `main` refers to the Python file containing your tabbyAPI app instance.
- `app` is the FastAPI instance defined in your Python script.
- `--host 0.0.0.0` allows access from external devices. Change this to `localhost` if you want to restrict access to the local machine.
- `--port 8000` specifies the port on which your application will run.
- `--reload` enables auto-reloading for development.

3. The tabbyAPI application should now be running. You can access it by opening a web browser and navigating to `http://localhost:8000` (if running locally).

## Usage

The tabbyAPI application provides the following endpoint:

- `/generate-text` (HTTP POST): Use this endpoint to generate text based on the provided input data.

### Example Request (using `curl`)


curl -X POST "http://localhost:8000/generate-text" -H "Content-Type: application/json" -d '{
    "model": "your_model_name",
    "messages": [
        {"role": "user", "content": "Say this is a test!"}
    ],
    "temperature": 0.7
}'
