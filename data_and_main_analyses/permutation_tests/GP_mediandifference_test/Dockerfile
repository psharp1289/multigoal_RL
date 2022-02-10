FROM python:3

ADD nonparametric_test_parameters_gp.py /
ADD requirements.txt /
ADD mf_key.json /
ADD gpd2.json /

RUN pip install -r requirements.txt

CMD [ "python", "./nonparametric_test_parameters_gp.py"]



