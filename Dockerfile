FROM debian:buster-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl

WORKDIR /opt
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Install required packages
RUN /opt/conda/bin/conda install -c defaults -c conda-forge numpy nibabel
ENV PATH /opt/conda/bin:$PATH
RUN pip install nipype

# Get python scripts
COPY phase_unwrap.py /opt

# Add volume mount point
VOLUME /data

ENTRYPOINT ["python", "/opt/phase_unwrap.py"]
