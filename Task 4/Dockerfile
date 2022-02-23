FROM python:3.8-slim
RUN apt-get update && apt-get install -y \
  xvfb python3-opengl ffmpeg \
  && rm -rf /var/lib/apt/lists/*
ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r /requirements.txt
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
WORKDIR /code
CMD xvfb-run -s "-screen 0 1400x900x24" python -u checker_client.py
