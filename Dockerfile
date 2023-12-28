# Use an official CUDA runtime with Ubuntu as a parent image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Get requirements
COPY requirements.txt requirements.txt

# Install torch with CUDA support and exllamav2
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install exllamav2

# Install any other needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run main.py when the container launches
CMD ["python3", "main.py"]
