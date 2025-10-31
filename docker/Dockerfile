# Use an official CUDA runtime with Ubuntu as a parent image
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    python3.12 \
    python3-pip \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the venv and set the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# set the working directory in the container
WORKDIR /app

# Get requirements
COPY pyproject.toml .

# Install packages specified in pyproject.toml cu12, extras
RUN pip install --no-cache-dir .[cu12,extras]

RUN rm pyproject.toml

# Copy the current directory contents into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# set the entry point
ENTRYPOINT ["python3"]

# Run main.py when the container launches
CMD ["main.py"]
