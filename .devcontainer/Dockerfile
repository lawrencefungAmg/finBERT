FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    pytorch_pretrained_bert numpy pandas nltk Flask flask-cors transformers \
    scikit-learn spacy textblob tqdm joblib ipykernel jupyter

RUN python -m nltk.downloader punkt stopwords
