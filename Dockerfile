FROM python:3.10.4
RUN apt update
RUN apt install git -y
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install scipy
RUN pip install python-decouple
RUN pip install ipykernel
RUN pip install pyswarms
