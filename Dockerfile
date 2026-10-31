FROM python:3.11-slim

WORKDIR /code

# Inference only needs these (no torch), keeping the image small
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart onnxruntime numpy pillow

COPY app ./app
COPY data ./data
COPY models ./models

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
