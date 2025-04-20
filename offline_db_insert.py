import argparse
from pathlib import Path
import re
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_args():
    parser = argparse.ArgumentParser(prog='Generate Data')
    parser.add_argument('--embedding_model', type=str, required=True, help='The Sentence Transformer Embedding Model')
    args = parser.parse_args()
    return args


def load_txt_files(folder_path: str):
    folder = Path(folder_path)
    dataset = {}
    for file in folder.glob('*.txt'):
        filename = os.path.basename(file)

        match = re.search(r"publish_date='(.*?)'_title='(.*?)'", filename)
        publish_date = match.group(1)
        title = match.group(2)

        dataset[(publish_date, title)] = file.read_text(encoding='utf-8')
    return dataset


def insert_db(embedding_model_name, no_pii, pii):

    # Create/Get Chroma DB & Initialize Embedding Model.
    client = chromadb.PersistentClient(path="./chroma_db")
    db = client.get_or_create_collection("blog_collection")
    embedding_model = SentenceTransformer(embedding_model_name)

    # Create the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # character count (not words)
        chunk_overlap=100    # overlap between chunks
    )

    # Helper function to process chunks
    def process(dataset, is_pii):
        for (publish_date, title), text in dataset.items():
            # Chunk the data
            chunks = text_splitter.split_text(text)

            # Add each chunk to Chroma
            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk)

                db.add(
                    ids=[f"{publish_date=}_{title=}_chunk={i}"],
                    documents=[chunk],
                    metadatas=[{
                        "title": title,
                        "publish_date": publish_date,
                        "chunk_id": i,
                        'is_pii': is_pii
                    }],
                    embeddings=[embedding]
                )

    process(no_pii, is_pii=False)
    process(pii, is_pii=True)


def main():
    # Get args
    args = parse_args()

    # Load data
    no_pii = load_txt_files('./data/No_PII/')
    pii = load_txt_files('./data/PII/')

    # Insert dataset into Chroma Vector DB
    insert_db(args.embedding_model, no_pii, pii)


if __name__ == "__main__":
    main()