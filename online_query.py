import argparse
from inference import Inference
import chromadb
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(prog='Generate Data')
    parser.add_argument('--model_name', type=str, required=True, help='The HuggingFace Model-Name or Model-Path')
    parser.add_argument('--embedding_model', type=str, required=True, help='The Sentence Transformer Embedding Model')
    args = parser.parse_args()
    return args

def load_model_and_db(model_name, embedding_model_name):
    system_message = """You are a helpful assistant trained to identify the presence of Personally Identifiable Information (PII) in datasets.
Your task is to analyze the given text and determine if it contains any form of PII, such as names, email addresses, phone numbers, social security numbers, addresses, credit card information, or other identifiable details.
Return a clear assessment:

"PII Detected" along with a brief explanation of what type(s) of PII were found

or "No PII Detected" if the data is clean.
Do not make assumptions. Only base your response on the content provided."""

    system_message = """You are tasked with identifying Personally Identifiable Information (PII) in datasets. PII refers to any information that can be used to identify a person, such as full names, addresses, phone numbers, email addresses, social security numbers, passport numbers, or any other data that could reasonably lead to identifying an individual. """

    # Load existing Chroma DB
    client = chromadb.PersistentClient(path="./chroma_db")
    db = client.get_collection("blog_collection")
    embedding_model = SentenceTransformer(embedding_model_name)

    # Initialize Pipeline
    inf = Inference(model_name=model_name, 
                    system_message=system_message, 
                    tp_size=1,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=500,
                    db=db,
                    embedding_model=embedding_model
                )
    return inf

def main():
    # Get args
    args = parse_args()

    # Load chroma-db, embedding model, and inference pipeline.
    inf = load_model_and_db(args.model_name, args.embedding_model)
    
    # Perform RAG and validate claims on sample queries
    claim = [{
        'where': ('meet miss sofia the fake travel blogger', '2025-03-23'), # Metadata (article publish_date, article title)
        'why': 'The combination of a full name, email, phone number, and city (New York City) could easily lead to the identification of Ava Jones',
        'how': "The blog clearly mentions Ava's full name, email, phone number, and city",
    }]

    outputs = inf.claims_query(claim)

    # Print output
    for i in range(0, len(outputs)):
        output = outputs[i].outputs[0].text.strip()
        print(f'{output=}')


if __name__ == "__main__":
    main()