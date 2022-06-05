FROM tensorflow/tensorflow:latest

ARG WD=/payload
WORKDIR ${WD}/
COPY peakfinder_final_model ${WD}/peakfinder_final_model

COPY requirements.txt ${WD}/requirements.txt
RUN pip install -r requirements.txt

COPY app.py ${WD}/app.py

CMD ["python", "app.py", "--port=5000"]
