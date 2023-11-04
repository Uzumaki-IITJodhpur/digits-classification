FROM python:3.8
WORKDIR /app
COPY experiments_modular.py /app/
COPY utils.py /app/
COPY requirements.txt /app/
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
VOLUME /app/saved_model
CMD ["python", "experiments_modular.py"]