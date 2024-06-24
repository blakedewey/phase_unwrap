FROM python:3.12.4-slim-bookworm

# Install required packages
RUN pip install nibabel numpy

# Install python package
COPY phase_unwrap /tmp

RUN pip install /tmp/phase_unwrap

ENTRYPOINT ["phase-unwrap"]