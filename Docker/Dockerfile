FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        curl wget ca-certificates sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        boto3 \
        xarray \
        netCDF4 \
        scipy \
        cfgrib \
        pyyaml \
        numpy \
        pandas \
        eccodes

WORKDIR /opt/sprite
