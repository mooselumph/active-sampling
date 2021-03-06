FROM nvidia-jupyter:latest

USER root

RUN ln -s /usr/local/cuda /usr/local/nvidia

RUN apt-get update && apt-get install -y git

RUN python -m pip install \
     tensorflow==2.3.0 \
     tensorflow-datasets \
     jax \
     jaxlib==0.1.69+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN python -m pip install flax

COPY requirements.txt .
RUN python -m pip --no-cache-dir install -r requirements.txt

USER $NB_USER
