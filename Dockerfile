FROM tensorflow/tensorflow:2.9.3

#RUN apt-get install git-lfs

COPY requirements.txt requirements.txt

RUN  pip install --use-feature=2020-resolver -r requirements.txt 

COPY aurore aurore