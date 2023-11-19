# Use an official CUDA runtime with Ubuntu as a parent image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Define a build-time argument for conditional installation
ARG INSTALL_FSCHAT=false

# Set the environment variable based on the build argument
ENV INSTALL_FSCHAT=$INSTALL_FSCHAT

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install torch with CUDA support and exllamav2
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install exllamav2

# Install any other needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Conditional installation of fschat[model_worker]
RUN if [ "$INSTALL_FSCHAT" = "true" ] ; then pip install fschat[model_worker] ; fi

# Copy the sample config file to the main config
RUN cp config_sample.yml config.yml

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["python3", "main.py"]
