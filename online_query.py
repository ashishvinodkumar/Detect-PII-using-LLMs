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
    system_message = """You are an expert PII detector. Reply with "Yes" if PII is found or "No" if it is not found, and briefly list the PII if found"""
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

    claims = [
        # PII - True Positive Example
        {
        'where': ('meet miss sofia the fake travel blogger', '2025-03-23'), # Metadata (article publish_date, article title)
        'why': 'The combination of a full name, email, phone number, and city (New York City) could easily lead to the identification of Ava Jones',
        'how': "The blog clearly mentions Ava's full name, email, phone number, and city",
        },

        # No PII - False Positive Example
        {
        'where': ('5 mustvisit destinations in europe for a memorable trip', '2025-03-24'), # Metadata (article publish_date, article title)
        'why': 'The combination of a full name, email, phone number, and city (New York City) could easily lead to the identification of Ava Jones',
        'how': "The blog clearly mentions Ava's full name, email, phone number, and city",
        },
    ]

    outputs = inf.claims_query(claims)

    # Print output
    for i in range(0, len(outputs)):
        output = outputs[i]
        print(f'{output=}')


if __name__ == "__main__":
    main()