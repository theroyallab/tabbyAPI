# Use an official CUDA runtime with Ubuntu as a parent image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG GIT_REPO=https://github.com/theroyallab/tabbyAPI
ARG DO_PULL=true
ENV DO_PULL $DO_PULL

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update repo
RUN if [ ${DO_PULL} ]; then \
    git init && \
    git remote add origin $GIT_REPO && \
    git fetch origin && \
    git pull origin main && \
    echo "Pull finished"; fi

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Get requirements
COPY pyproject.toml .

# Install packages specified in pyproject.toml cu121
RUN pip3 install --no-cache-dir .[cu121]

# Copy the current directory contents into the container
COPY . .

# Create a config.yml
COPY config_sample.yml config.yml

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set the entry point
ENTRYPOINT ["python3"]

# Run main.py when the container launches
CMD ["main.py"]
