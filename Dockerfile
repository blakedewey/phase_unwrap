FROM python:3.12.4-slim-bookworm

COPY . /tmp/phase_unwrap-src

RUN apt update && \
    apt install -y --no-install-recommends ca-certificates git && \
    pip install --no-cache-dir /tmp/phase_unwrap-src/ && \
    rm -rf /tmp/phase_unwrap-src/ && \
    apt remove -y git && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["unwrap-phase"]