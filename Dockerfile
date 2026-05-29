# Base image — PyTorch 2.3.1 with CUDA 12.1 and cuDNN 8, pre-built by the PyTorch team.
# This gives us: Python 3.10, torch, torchvision, CUDA toolkit, cuDNN.
# We add our project's Python dependencies on top.
#
# IMPORTANT: Do NOT copy source code into this image.
# Condor transfers your scripts to the compute node at job submission time and
# mounts them into the container's working directory automatically.
# The image only needs to contain the *environment* — Python packages and system libs.
# This means you can update your training code without rebuilding the image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set the working directory inside the container.
# Condor will mount your transferred files here when the job runs.
WORKDIR /workspace

# Copy only the requirements file into the image.
# We do this as a separate step from the pip install so Docker can cache the
# install layer — if requirements.txt hasn't changed, Docker skips the pip step
# on the next build, which saves several minutes.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: don't store the pip download cache inside the image layer,
#                 keeps the image smaller.
# --upgrade pip:  ensures we're using a recent pip before installing packages.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
