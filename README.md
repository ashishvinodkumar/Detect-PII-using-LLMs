# Detect PII with a RAG based LLM

### Breif Overview of Approach

To goal of this project is to implement RAG with Context Relevance, Groundedness, and Answer Relevance, for effectively identifying PII data in any given dataset.

Context Relevance
The first step of any RAG application is retrieval; to verify the quality of our retrieval, we want to make sure that each chunk of context is relevant to the input query to prevent hallucination. 

Groundedness
After the context is retrieved, it is then formed into an answer by an LLM. LLMs are often prone to stray from the facts provided, exaggerating or expanding to a correct-sounding answer. To verify the groundedness of our application, we can separate the response into individual claims and independently search for evidence that supports each within the retrieved context.

Answer Relevance
Last, our response still needs to helpfully answer the original question. We can verify this by evaluating the relevance of the final response to the user input.


### How?

To implement this RAG LLM solution for PII clasificaiton, we effectively need an offline and online process. 

#### Offline Backend Batch Workflow:
- Each of the "Data Providers" that upload their datasets onto the platform, need to not only be catalogued for future consumption, but also ingested into a Vector Database to enable Retrieval Augmented Generation (RAG).
- The 'offline_insert.py' script parses through all available datasets, extracts metadata and prepares it for database insertion.
- Furthermore, the offline process also includes chunking and embedding the data into Chroma DB's vector database with metadata centered around 'title' and 'publish_data' for faster and accurate retrieval. 
- The offline workflow will run once every few hours to continually index new datasets as and when they are uploaded.

#### Online Realtime Workflow:
- If/When a 'Data Consumer' identifies PII data that they want to flag. They will interface with a portal to provide the following inputs that formulate a 'claim':
    1. Where:
    2. When:
    3. How:
    4. Source Title:
    5. Publish Data:
- Once the following information is entered, this 'claim' is captured by the 'online query.py' script that performs Retrieval Augmented Generation (RAG) to retrieve the top-n most relevant chunks supported by the query.
- Specifically, based on the claim, we filter the vector db by the underlying source title and publish date metadata. This helps narrow down search space and improve accuracy. The top-n relevant chunks from the reduced search space is then processed to create a 'context' dataset. 
- This context dataset is then passed onto the TinyLlama 1b model to arbitrate whether there is PII present in the dataset or not.


### Steps to Run
1. Create Python Virtual Environment and Install Packages
```shell
make venv
make install
```

2. Synthesize fake PII and non-PII data.
```shell
make data
```

3. Create Persistent Vector DB
```shell
make offline_insert
```

4. Query the Vector DB
```shell
make only_query
```

### Obvious Flaws/Next Steps

I put together this application in a few hours on a Macbook Air with no GPUs, so I had to limit myself to a TinyLlama 1 Billion Parameter model. Having previously built many workflows like this for various use-cases, I can confidently say that bumping up the LLM from a TinyLlama 1B to Meta's Llama3.1 8B (or other such model), which requires a single Nvidia A100 GPU, will yield incredible results for a PII classification problem with LLMs.

Furthermore, TinyLlama cannot effectively perform Chain-Of-Thought (COT) reasoning or Few-Shot-Learning, which is necessary for effectively categorizing responses in a streamlined binary output. As mentioned before, using Meta's Llama3.1 8B model with Few-Shot-Learning will be the ideal next step. We can also further control the response generation by using JSON based response guidelines, to enforce a specific formatted response such as {"PII": True/False, "Reasoning/Evidence": ''}

Is there an effective chunking strategy? The short answer is NO. However, one can make educated decisions based on the distribution of the underlying data to chunk meaningful datasets together with setting a generous text overlap. For this project, I mocked up data using TinyLlama 1B, which always repeatedly adds 'fictional/fake' identifiers before and after adding any PII data (for safety reasons). This makes chunking and effective response generation on synthetic data for PII identification a little tricky. 

Next, s there a plan for scaling this solution? Yes, the 'Inference' class can easily be wrapped inside multiple Python MultiProcessing workers, that are always active in a while True loop, until torn down. A next step here would be to wrap the online_query's main function in the following architecture.

```
import multiprocessing as mp

# The target='inference' below will internally call the existing 'inference.py' class.

input_queue = mp.Queue()
response_queue = mp.Queue()
N=4
procs = [mp.Process(target=inference, args=(input_queue, response_queue)) for _ in range(0, N)]
procs.start()
input_queue.put([claim1, claim2, claim3, ...])
outputs = response_queue.get()
```

Finally, How do we go about evaluating performance and accuracy? 

1. For context relevance, most Vector Databases like Qdrant and ChromaDB have a built-in Consine Similarity. We can evaluate the minimum needed metadata to consistently obtain the most rich vectors/context from the database. We can also experiment with other approaches such as Manhattan distance, Euclidian distance, Inner Product, and more to identify which metric provides the most optimal result. 

2. For Groundedness, first we can use Prompt Engineering to force groundedness in the response by Few-Shot-Learning examples and explicitly enforcing LLM not use any information outside of the context provided, in generating a response. Furthermore, we can also use LLM-As-A-Judge with a more superior LLM to measure distribution in response across various LLMs, or we can calculate the faithfulness and context precision in relation to the underlying context. 

3. For Answer Relevance, we can similarly measure the cosine similarity measure with a minimum threshold bar for 'correctness'. We can also use lightweight NLP techniques that count word affinity in response to query for additional validation. 