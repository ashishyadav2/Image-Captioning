FROM python:3.11-slim-bullseye
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python setup.py build
RUN python setup.py install
RUN python src/components/data_ingestion.py
EXPOSE 5000
CMD python app.py