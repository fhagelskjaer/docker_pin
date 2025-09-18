FROM python:3.6

    
RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY ws_cad_dl .

RUN pip install .

CMD ["python3", "scripts/test_network.py"]

