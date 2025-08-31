FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "--login", "-c"]
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip cmake g++ gcc make build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev

RUN cd / && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    /bin/bash -c "source ~/.bashrc" && \
    /opt/conda/bin/conda update -n base -c defaults conda -y &&\
    /opt/conda/bin/conda create -n Any6D python=3.9

ENV PATH $PATH:/opt/conda/envs/Any6D/bin
COPY . .

RUN conda init bash &&\
    echo "conda activate Any6D" >> ~/.bashrc &&\
    conda activate Any6D &&\
    conda install conda-forge::eigen=3.4.0 &&\
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/opt/conda/envs/Any6D" &&\
    python -m pip install -r requirements.txt &&\
    python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git &&\
    FORCE_CUDA=1 python -m pip install --no-cache-dir kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html &&\
    pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu121

# this is needed to link where conda installs eigen3 vs where it is expected
RUN ln -s /opt/conda/envs/Any6D/include/eigen3 /usr/local/include/eigen3

RUN conda init bash &&\
    conda activate Any6D &&\
    export CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 &&\
    cd foundationpose/mycpp && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11 &&\
    cd ../.. &&\
    cd bundlesdf/mycuda && rm -rf build *egg* && pip install -e . &&\
    cd .. && cd ..

RUN conda init bash &&\
    conda activate Any6D &&\
    cd sam2 && pip install -e . && cd ..

RUN conda init bash &&\
    conda activate Any6D &&\
    cd instantmesh && pip install -r requirements.txt && cd ..

RUN conda init bash &&\
    conda activate Any6D &&\
    cd bop_toolkit && python setup.py install && cd ..

RUN conda init bash &&\
    conda activate Any6D &&\
    pip install huggingface-hub==0.24.0

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh
