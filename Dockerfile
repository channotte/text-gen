FROM tensorflow/tensorflow:latest

RUN apt-get install git-lfs

RUN  pip install -r requirements.txt