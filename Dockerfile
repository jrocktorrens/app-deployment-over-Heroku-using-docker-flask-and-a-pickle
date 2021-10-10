FROM python:3.8
WORKDIR /opt
ADD dockerized_app/ /opt
RUN pip install -r requirements.txt
EXPOSE 5000
ENV FLASK_APP=inference.py
ENTRYPOINT flask run --host 0.0.0.0 -p 5000
