FROM python:3.12.4-slim-bookworm

# Install required packages
RUN pip install nibabel numpy

# Install python package
COPY . /tmp/phase_unwrap-src

RUN pip install --no-cache-dir /tmp/phase_unwrap-src/ && \
    rm -rf /tmp/phase_unwrap-src/

ENTRYPOINT ["phase-unwrap"]