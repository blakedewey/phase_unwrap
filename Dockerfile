FROM python:3.12.4-slim-bookworm

# Install required packages
RUN pip install nibabel numpy

# Get python scripts
COPY phase_unwrap.py /opt

ENTRYPOINT ["python", "/opt/phase_unwrap.py"]