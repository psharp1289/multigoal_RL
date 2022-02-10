FROM python:3

ADD permutation_test_changeparams.py /
ADD requirements.txt /
ADD mf_key.json /
ADD chparam.json /

RUN pip install -r requirements.txt

CMD [ "python", "./permutation_test_changeparams.py"]



