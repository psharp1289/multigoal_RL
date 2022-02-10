FROM python:3

ADD nonparametric_test_parameters_mb.py /
ADD requirements.txt /
ADD mf_key.json /

RUN pip install -r requirements.txt

CMD [ "python", "./nonparametric_test_parameters_mb.py"]



