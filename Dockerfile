FROM python:3.8-slim

COPY ./ensembles.py /root/server/src/ensembles.py
COPY ./src/requirements.txt /root/server/src/requirements.txt

RUN chown -R root:root /root/server

WORKDIR /root/server/src
RUN pip3 install -r requirements.txt

COPY src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY tree
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]
