FROM nvidia/cuda:11.8.0-base-ubuntu22.04
# FROM ubuntu:20.04

# To avoid tk-dev asking for geographic location...
ENV DEBIAN_frontend noninteractive
ARG DEBIAN_FRONTEND=noninteractive

# Used for defining the expected sha256 checksum
ENV EXPECTED_SHA256_CONDA=6c8a4abb36fbb711dc055b7049a23bbfd61d356de9468b41c5140f8a11abd851

# Just installing python3 for now, will worry about 3.7.13 later
RUN apt-get update -y && apt-get upgrade -y
RUN apt install -y curl
RUN curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
RUN echo "$EXPECTED_SHA256_CONDA *anaconda.sh | sha256sum -c -"
RUN bash anaconda.sh -b -p /opt/conda

# Adds anaconda bin's to the path so we can use commands like "conda"
ENV PATH=/opt/conda/bin:$PATH

RUN apt install -y cuda-toolkit-11-8
RUN apt install -y g++-11

# 
WORKDIR /
COPY environment.yml .
COPY /submodules ./submodules
RUN conda env create --file environment.yml
RUN conda init bash && exec bash && conda activate gaussian_splatting
# At this point we now have conda installed. 

# getting SIBR viewer
# Install necessary packages
RUN apt install -y git && \
    apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev 
    #&& \
    # apt clean && apt install wget && rm -rf /var/lib/apt/lists/*
    
RUN apt update
RUN apt install -y cmake build-essential

RUN apt install -y libcurl4-openssl-dev libgdal-dev

# Create a workspace directory and clone the repository
# We use this repository because I made changes that allow us to set the 
# number of iterations to stop training at through terminal arguments
RUN git clone https://github.com/JoshuaG-K/gaussian-splatting --recursive

# Tweak the CMake file for matching the existing OpenCV version. Fix the naming of FindEmbree.cmake
WORKDIR /gaussian-splatting/SIBR_viewers/cmake/linux
RUN sed -i 's/find_package(OpenCV 4\.5 REQUIRED)/find_package(OpenCV 4.2 REQUIRED)/g' dependencies.cmake
RUN sed -i 's/find_package(embree 3\.0 )/find_package(EMBREE)/g' dependencies.cmake
RUN mv /gaussian-splatting/SIBR_viewers/cmake/linux/Modules/FindEmbree.cmake /gaussian-splatting/SIBR_viewers/cmake/linux/Modules/FindEMBREE.cmake

# Fix the naming of the embree library in the rayscaster's cmake
RUN sed -i 's/\bembree\b/embree3/g' /gaussian-splatting/SIBR_viewers/src/core/raycaster/CMakeLists.txt

WORKDIR /gaussian-splatting
COPY /data ./data
RUN apt install -y nvidia-container-toolkit
RUN apt install -y colmap
# CMD ["conda init bash && exec bash"]

# RUN unset LD_LIBRARY_PATH

# Ready to build the viewer now.
# WORKDIR /gaussian-splatting/SIBR_viewers 
# RUN cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
#    cmake --build build -j24 --target install


