FROM python:3.7

WORKDIR /app/scripts

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

COPY . /app

CMD python train_logistic_model.py