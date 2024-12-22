###############################################################################
#
# Test to see if your vision model is working
#
# Remember to set vision: true for your model in config.yml
#
###############################################################################
import base64
import io
import json
import os
import requests
from mimetypes import MimeTypes

import climage
from PIL import Image

if os.environ.get("X_API_KEY", False):
    X_API_KEY = os.environ["X_API_KEY"]
else:
    print("You must pass in an environment variable for the API key.")
    exit(f"ex: X_API_KEY=123456789abcdef python {os.path.basename(__file__)}")

instructions = "Compare and contrast the two experiments."
images = [
    {"file": "test_image_1.jpg"},
    {"file": "test_image_2.jpg"},
    # {"url": "https://media.istockphoto.com/id/1212540739/photo/mom-cat-with-kitten.jpg?s=612x612&w=0&k=20&c=RwoWm5-6iY0np7FuKWn8FTSieWxIoO917FF47LfcBKE="},
    # {"url": "https://i.dailymail.co.uk/1s/2023/07/10/21/73050285-12283411-Which_way_should_I_go_One_lady_from_the_US_shared_this_incredibl-a-4_1689019614007.jpg"},
    # {"url": "https://images.fineartamerica.com/images-medium-large-5/metal-household-objects-trevor-clifford-photography.jpg"}
]


def get_cli_images(images):
    cli_images = list()
    for image in images:
        for k, v in image.items():
            if k == "file":
                script_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(script_dir, v)
                output = climage.convert(file_path, is_unicode=True, width=50)
                cli_images.append(output)
            if k == "url":
                response = requests.get(v)
                img = Image.open(io.BytesIO(response.content))
                output = climage.convert_pil(img, is_unicode=True, width=50)
                cli_images.append(output)
    return cli_images


def uri_encode_image(location_type, value):
    if location_type == "url":
        return value
    if location_type == "file":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, value)
        mime = MimeTypes()
        mime_type = mime.guess_type(file_path)[0]
        with open(file_path, "rb") as image_file:
            data = base64.b64encode(image_file.read())
            image_uri = "data:" + mime_type + ";" + "base64," + str(data)[2:-1]
            return image_uri
    raise TypeError("Unsupported image location type")


def check_model():
    response = requests.get(
        "http://127.0.0.1:5000/v1/model",
        headers={
            "Content-Type": "application/json",
            "x-api-key": X_API_KEY
        }
    )
    model_info = response.json()
    for k, v in model_info.items():
        if k not in ["logging", "parameters"]:
            print(f"{k}: {v}")
        else:
            for k2, v2 in model_info[k].items():
                print(f"{k2}: {v2}")
    return model_info


def tabbyapi_server(messages, max_tokens=500, temperature=0.0):
    response = requests.post(
        "http://127.0.0.1:5000/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "x-api-key": X_API_KEY
        },
        data=json.dumps({
            "model": "Qwen2-VL-7B-Instruct-exl2-8_0BPW",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        })
    )
    return response.json()


def get_messages(instructions, images):
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
            ],
        }
    ]
    for image in images:
        for k, v in image.items():
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": uri_encode_image(k, v),
                },
            }
            messages[0]["content"].append(image_content)
    return messages


### MAIN ###
try:
    model_info = check_model()
    if not model_info["parameters"]["use_vision"]:
        exit("FAILURE: The active model does not support vision")
except requests.exceptions.ConnectionError:
    exit("FAILURE: tabbyAPI server is not active.")
print(f"USER: {instructions}")
for cli_image in get_cli_images(images):
    print(cli_image)
output = tabbyapi_server(get_messages(instructions, images))
print(f'VISION ASSISTANT: {output["choices"][0]["message"]["content"]}')
