FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6
RUN mkdir /logs
RUN mkdir /data
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8 LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN sed -i 's/archive.ubuntu.com/ftp.daumkakao.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y wget nano git gcc build-essential
RUN apt-get install -y software-properties-common

RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN add-apt-repository -y ppa:ubuntugis/ppa && apt-get update
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal

COPY requirements.txt /fastapi/requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

COPY . /app/

#CMD ["python3", "main.py"]
