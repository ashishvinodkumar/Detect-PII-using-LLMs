import argparse
from inference import Inference
from datetime import datetime, timedelta
import random 
import re

def parse_args():
    parser = argparse.ArgumentParser(prog='Generate Data')
    parser.add_argument('--model_name', type=str, required=True, help='The HuggingFace Model-Name or Model-Path')
    parser.add_argument('--num_articles', type=int, required=True, help='The number of artices you want generated.')
    args = parser.parse_args()
    return args

def get_prompts(num_articles: int=500):
    prompts = []
    for i in range(0, 2*num_articles):
        if i % 2 == 0: # With PII
            prompt = 'Write a short 500 word travel blog with a short title, describing a fictional person including their name, age, address, phone number, and email. Make the information sound realistic but entirely fake.'
        else: # Without PII
            prompt = 'Write a short 500 word travel blog with a short title, summary that includes places visited, and travel modes. Avoid using any personal names or identifying information.'
        prompts.append(prompt)
    return prompts

def main():
    # Get args
    args = parse_args()

    # Get synthetic data
    system_message = "You are an online blog specialist that writes travel blogs. Be clear and concise."
    prompts = get_prompts(num_articles=args.num_articles)
    batch_size = 4
    batches = [prompts[i:i+batch_size] for i in range(0, 2*args.num_articles, batch_size)]

    # Initialize Pipeline
    inf = Inference(model_name=args.model_name, 
                    system_message=system_message, 
                    tp_size=1,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=500,
                )
    
    # Create .txt synthetic data files
    for i in range(0, len(batches)):
        prompts = batches[i]
        outputs = inf.vanilla_query(prompts)
        write_to_txt(outputs, batch_id=i)

def write_to_txt(outputs: list, batch_id: str):
    # Parse response and create synthetic blog articles
    for i in range(0, len(outputs)):
        output = outputs[i].outputs[0].text.strip()

        # Generate a random publish-date for metadata
        now = datetime.now() - timedelta(days=random.randint(1, 30))
        publish_date = now.strftime("%Y-%m-%d")

        # Extract time for metadata.
        title = output.splitlines()[0].lower().replace('title:', '')
        title = title[0:200].strip()
        title = re.sub(r"[^A-Za-z0-9 ]+", "", title)

        if i % 2 == 0: # With PII
            filename = f'./data/PII/{publish_date=}_{title=}.txt'
        else: # Without PII
            filename = f'./data/No_PII/{publish_date=}_{title=}.txt'

        with open(filename, "w") as text_file:
            text_file.write(output)

if __name__ == "__main__":
    main()