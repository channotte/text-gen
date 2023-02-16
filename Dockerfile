FROM tensorflow/tensorflow:2.9.3

COPY requirements.txt requirements.txt

RUN  pip install --use-feature=2020-resolver -r requirements.txt 

EXPOSE 8080  

COPY aurore aurore