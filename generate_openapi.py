import json
from endpoints import server


if __name__ == "__main__":
    """Uses the FastAPI server to write an OpenAPI JSON documentation file."""

    server.setup_app()
    openapi_json = json.dumps(server.app.openapi())

    # Write JSON to a file
    with open("openapi.json", "w") as f:
        f.write(openapi_json)
