from vllm import LLM, SamplingParams
import torch
 
class Inference:
    def __init__(self, 
                 model_name, 
                 system_message, 
                 tp_size,
                 temperature,
                 top_p,
                 max_tokens,
                 db=None,
                 embedding_model=None):
        
        self.model_name = model_name
        self.system_message = system_message
        self.tp_size = tp_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.db = db
        self.embedding_model=embedding_model

        # Load Model
        self.llm, self.sampling_params = self.load_model()

    def load_model(self):
        use_mps = torch.backends.mps.is_available()
        device_type = "mps" if use_mps else "cpu"
        print(f"Using device: {device_type}")

        # Initialize LLM
        llm = LLM(model=self.model_name, 
                download_dir="./models",
                tensor_parallel_size=self.tp_size,
                trust_remote_code=True)
        
        # Set sampling parameters
        sampling_params = SamplingParams(temperature=self.temperature, 
                                        top_p=self.top_p, 
                                        max_tokens=self.max_tokens)
        
        return llm, sampling_params
    

    def format_prompt(self, user_message: str, history: list = None):
        prompt = f"<|system|>\n{self.system_message}\n"
        if history:
            for turn in history:
                prompt += f"<|user|>\n{turn['user']}\n<|assistant|>\n{turn['assistant']}\n"
        prompt += f"<|user|>\n{user_message}\n<|assistant|>\n"
        return prompt
    

    def vanilla_query(self, prompts):
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        outputs = self.llm.generate(formatted_prompts, 
                                    sampling_params=self.sampling_params)
        return outputs
    
    def print_all_vectordb(self):
        results = self.db.get(limit=1000, include=["metadatas"])
        for doc_id, metadata in zip(results["ids"], results["metadatas"]):
            print(f"ID: {doc_id}")
            print("Metadata:", metadata)
            print("-" * 40)

    def get_relevant_chunks(self, title, publish_date, issue, instance):
        query_embedding = self.embedding_model.encode(str(issue + instance))
        filters = {
                    "$and": 
                        [
                            {
                                "publish_date": {
                                    '$eq': publish_date
                                }
                            },
                            {
                                "title": {
                                    '$eq': title
                                }
                            }
                        ]
                    }
        
        results = self.db.query(
            query_embeddings=[query_embedding],
            n_results=2,  # Retrieve top 3 most relevant blog posts
            where=filters,
        )
        retrieved_text = '\n\n'.join(results['documents'][0])
        return retrieved_text

    def claims_query(self, claims):
        for claim in claims:
            title = claim['where'][0]
            publish_date = claim['where'][1]
            issue = claim['why']
            instance = claim['how']

            retrieved_text = self.get_relevant_chunks(title, publish_date, issue, instance)
            print(f'{retrieved_text=}')

            # Create a prompt with the retrieved context for RAG
            prompt = f"Here is the datasets to check for PII:\n{retrieved_text}\n\nPlease analyze and indicate if there is any PII present in this dataset.'"
            formatted_prompts = self.format_prompt(prompt)
            print(f'\n\n{formatted_prompts=}\n\n')

            # Use vLLM for inference
            outputs = self.llm.generate(formatted_prompts, 
                                        sampling_params=self.sampling_params)
            return outputs