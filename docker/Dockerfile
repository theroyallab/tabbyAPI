# Use an official CUDA runtime with Ubuntu as a parent image
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Install system dependencies. python3.12-dev supplies Python.h, which Triton's
# runtime JIT needs to build its cuda_utils extension inside the container. The
# venv resolves its include path to /usr/include/python3.12, so the headers have
# to come from the system python, not from the venv.
# See: https://github.com/theroyallab/tabbyAPI/issues/453
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the venv and set the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Get requirements
COPY pyproject.toml .

# Install cu12 group — pins torch+cu128, exllamav3+cu128,
# and flash-linear-attention.
RUN pip install --no-cache-dir .[cu12]

# Optionally install the embeddings stack (infinity-emb, sentence-transformers)
# with --build-arg EXTRAS=1 (disabled by default).
# The installed torch is pinned via a constraints file so dependency resolution
# cannot replace it with a wheel built for a different CUDA version, which would
# cause libcudart ImportErrors on hosts with older drivers.
# See: https://github.com/theroyallab/tabbyAPI/issues/414
ARG EXTRAS=
RUN if [ -n "$EXTRAS" ]; then \
      pip freeze | grep -E '^torch(==| @)' > /tmp/torch-constraint.txt && \
      pip install --no-cache-dir -c /tmp/torch-constraint.txt .[extras]; \
    fi

RUN rm pyproject.toml

# Copy the current directory contents into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set the entry point
ENTRYPOINT ["python3"]

# Run main.py when the container launches
CMD ["main.py", "--host", "0.0.0.0"]
