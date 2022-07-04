FROM rapidsai/rapidsai-core:22.06-cuda11.5-base-ubuntu20.04-py3.8

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY ./ ./

RUN apt update && apt -y upgrade && apt install -y \
  ca-certificates \
  build-essential \
  cmake \
  git \
  gcc \
  g++ \
  curl \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev

RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install

# Install LightGBM
RUN conda install -q -y numpy scipy scikit-learn pandas && \
  git clone --recursive --branch stable --depth 1 https://github.com/Microsoft/LightGBM && \
  cd LightGBM/python-package && python setup.py install && \
  apt-get autoremove -y && apt-get clean && \
  conda clean -a -y && \
  rm -rf /usr/local/src/*

CMD bash
