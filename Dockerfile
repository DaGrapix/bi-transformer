FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    git \
    python3 \
    python3-pip

# Copy all the files to the working directory
COPY ./src/ src/

# Get the LIPS repository
RUN git clone https://github.com/IRT-SystemX/LIPS.git

# Remove scipy and numpy from the setup file to avoid conflicts with system packages
COPY setup.py LIPS/setup.py

# Install LIPS
RUN cd LIPS && pip install -U . --break-system-packages && cd ..

# Install airfrans
RUN pip install airfrans --break-system-packages

# Install PyTorch
RUN pip install torch -f https://download.pytorch.org/whl/cu121 --break-system-packages
RUN pip install torch_geometric --break-system-packages

# Download the dataset
COPY get_dataset.py get_dataset.py
RUN python3 get_dataset.py
RUN rm Dataset.zip
