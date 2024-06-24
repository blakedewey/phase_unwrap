FROM python:3.12.4-slim-bookworm

RUN apt update && \
    apt install -y --no-install-recommends ca-certificates git

# Install required packages
RUN pip install nibabel numpy

# Install python package
COPY . /tmp/phase_unwrap-src

RUN pip install --no-cache-dir /tmp/phase_unwrap-src/ && \
    rm -rf /tmp/phase_unwrap-src/

ENTRYPOINT ["phase-unwrap"]