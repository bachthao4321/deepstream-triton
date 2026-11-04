FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# System dependencies required by DeepStream 9.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GStreamer
    libgstreamer1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-rtsp \
    # Display / graphics
    libcairo2 \
    libpangocairo-1.0-0 \
    libx11-6 \
    libgbm1 \
    libglapi-mesa \
    libgles2-mesa-dev \
    # Utilities
    libglib2.0-0 \
    libyaml-cpp-dev \
    # Python
    python3-pip \
    python3-dev \
    python3-venv \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    # TensorRT 10.x
    libnvinfer10 \
    libnvinfer-dev \
    libnvinfer-headers-dev \
    libnvinfer-plugin10 \
    libnvinfer-plugin-dev \
    libnvonnxparsers10 \
    libnvonnxparsers-dev \
    # CUDA 13.1 components (satisfies DeepStream depends: cuda-cudart-13-1)
    cuda-cudart-13-1 \
    cuda-cudart-dev-13-1 \
    libnpp-13-1 \
    libnpp-dev-13-1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install DeepStream 9.0
COPY deepstream-9.0_9.0.0-1_amd64.deb /tmp/deepstream.deb
RUN dpkg -i /tmp/deepstream.deb && rm /tmp/deepstream.deb

# Run DeepStream additional install (Triton backend, etc.)
RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh 2>/dev/null || true

# Triton Python client
RUN pip3 install --no-cache-dir --break-system-packages \
    tritonclient[all] \
    numpy \
    opencv-python-headless

ENV DS_VERSION=9.0
ENV CUDA_VER=13.1
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
ENV PATH=/opt/nvidia/deepstream/deepstream/bin:$PATH

WORKDIR /workspace

CMD ["/bin/bash"]
