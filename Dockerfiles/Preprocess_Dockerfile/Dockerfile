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
# Installing conda will mess up the colmap installation below
#RUN curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
#RUN echo "$EXPECTED_SHA256_CONDA *anaconda.sh | sha256sum -c -"
#RUN bash anaconda.sh -b -p /opt/conda

# Adds anaconda bin's to the path so we can use commands like "conda"
#ENV PATH=/opt/conda/bin:$PATH

RUN apt update
RUN apt install -y cuda-toolkit-11-8
RUN apt install -y gcc-10 g++-10
RUN export CC=/usr/bin/gcc-10
RUN export CXX=/usr/bin/g++-10
RUN export CUDAHOSTCXX=/usr/bin/g++-10

# 
#WORKDIR /server
#COPY environment.yml .
#COPY /submodules ./submodules
#RUN conda env create --file environment.yml
#RUN conda init bash && exec bash && conda activate gaussian_splatting
# At this point we now have conda installed. 

# getting SIBR viewer
# Install necessary packages
RUN apt install -y git && \
    apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev 
    #&& \
    # apt clean && apt install wget && rm -rf /var/lib/apt/lists/*
    
RUN apt update
RUN apt-get clean
RUN apt install -y cmake build-essential

RUN apt install -y libcurl4-openssl-dev libgdal-dev

# Create a workspace directory and clone the repository
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# Tweak the CMake file for matching the existing OpenCV version. Fix the naming of FindEmbree.cmake
# WORKDIR /server/gaussian-splatting/SIBR_viewers/cmake/linux
# RUN sed -i 's/find_package(OpenCV 4\.5 REQUIRED)/find_package(OpenCV 4.2 REQUIRED)/g' dependencies.cmake
# RUN sed -i 's/find_package(embree 3\.0 )/find_package(EMBREE)/g' dependencies.cmake
#RUN mv /server/gaussian-splatting/SIBR_viewers/cmake/linux/Modules/FindEmbree.cmake /server/gaussian-splatting/SIBR_viewers/cmake/linux/Modules/FindEMBREE.cmake

# Fix the naming of the embree library in the rayscaster's cmake
# RUN sed -i 's/\bembree\b/embree3/g' /server/gaussian-splatting/SIBR_viewers/src/core/raycaster/CMakeLists.txt

WORKDIR /gaussian-splatting
COPY /data ./data
# RUN apt install -y nvidia-container-toolkit
# RUN apt install -y colmap
RUN apt-get update
RUN apt-get install -y \
   libboost-program-options-dev \
   libboost-filesystem-dev \
   libboost-graph-dev \
   libboost-system-dev \
   libboost-test-dev \
   libeigen3-dev \
   libsuitesparse-dev \
   libfreeimage-dev \
   libgoogle-glog-dev \
   libgflags-dev \
   libglew-dev \
   libcgal-dev

RUN git clone https://github.com/colmap/colmap.git
RUN apt-get install -y \
    ninja-build \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
    
RUN cd colmap

WORKDIR /gaussian-splatting/colmap/build

RUN cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 -GNinja
RUN ninja
RUN ninja install

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PATH="/root/.pyenv/bin:${PATH}"
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc

# Install the specific version of Python
ARG PYTHON_VERSION=3.7.13
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

WORKDIR /gaussian-splatting
# RUN pip install --upgrade pip

# RUN cmake .. -DCMAKE_CUDA_FLAGS="--keep --keep-dir=tmp -gencode=arch=compute_89,code=sm_89" -DCMAKE_CUDA_ARCHITECTURES=native -GNinja
# RUN cmake .. -DCUDA_ARCHITECTURES=89 -DCMAKE_CUDA_ARCHITECTURES=native -GNinja

# RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH="/user/local/x86-64-linux-gnu/cmake/Qt5" -DCMAKE_CXX_FLAGS="-std=c++11" -DCUDA_ENABLE="TRUE" -DBUILD_GUI="FALSE" -GNinja
	
#RUN ninja
#RUN ninja install

#RUN apt install -y libxcb-xinerama0
#RUN export QT_QPA_PLATFORM=offscreen
# CMD ["conda init bash && exec bash"]

# RUN unset LD_LIBRARY_PATH

# Ready to build the viewer now.
# WORKDIR /server/gaussian-splatting/SIBR_viewers 
# RUN cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
#    cmake --build build -j24 --target install


