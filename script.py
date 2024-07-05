from io import BytesIO
import os
from PyPDF2 import PdfReader
import dotenv
import numpy as np
import openai


def load_env():
    dotenv.load_dotenv()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    return endpoint, api_key, deployment

def set_azure_client(endpoint, api_key):
    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01"
    )
    return client

def extract_content_from_file(pdf_file):
    pdf = PdfReader(BytesIO(pdf_file.read()))
    content = ""
    for page in range(len(pdf.pages)):
        content += pdf.pages[page].extract_text()
    return content

def split_pdf_content(content, max_length=3000):
    words = content.split()
    segments = []
    current_segment = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_length:
            segments.append(''.join(current_segment))
            current_segment = [word]
            current_length = len(word) + 1
        else:
            current_segment.append(word)
            current_length += len(word) + 1
    
    if current_segment:
        segments.append(''.join(current_segment))
    
    return segments

def make_embeddings(all_text_segments, client):
    embeddings = []
    for segment in all_text_segments:
        response = client.embeddings.create(input=segment, model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)
    return embeddings

def search(query, embeddings, all_text_segments, client):
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding
    scores = [np.dot(query_embedding, emb) for emb in embeddings]
    best_match_index = np.argmax(scores)
    return best_match_index, all_text_segments[best_match_index]