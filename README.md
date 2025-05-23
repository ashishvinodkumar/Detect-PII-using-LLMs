# Detect PII with a RAG based LLM

### Problem Statement
BlogBot, an AI company, has identified PII data while training its GenAI model. They have reached out to us with a 'claim' comprising of where the PII data was observed. We need to validate this claim, i.e detect PII information in the underlying data, and confirm/deny the claim. Beyond the scope of this project: Resolve/Remove the PII conflicts within the dataset. 

### 3 Approaches to Potentially Solve the Problem
1. Fine-Tune an LLM to accurately identify PII data.
    - This solution requires at least 2-4 GPUs depending on the scale and size of the LLM that needs to be fine-tuned.
    - This is a cost ineffective option, time-consuming, and not the right first step, especially since an Inference workflow has not yet been attempted.
    - Only if there is a nuanced scenario that a general purpose LLM with Chain-of-Thought reasoning is unable to perform, will we consider fine-tuning an LLM.
2. Utilize only an LLM with large context length for real-time PII detection.
    - While this approach could work, maintaining large contexts could give rise to hallucination, and is also cost inefficient as large context lenghts would require more compute and/or GPUs.
    - Most documents are large (GBs/TBs), so it is also computationally inefficient to store such large files in memory if the frequency of retrieval is not every few hours.
3. Build a RAG based LLM Inference workflow with Prompt Engineering.
    - This is an ideal first step, as we can get reasonably good performance and accuracy with identifying PII information in the underlying dataset without training any LLMs on expensive GPUs.
    - Requires a deep understanding of Chain-Of-Thought (COT) reasoning, Few-Shot-Learning, Prompt Engineering, distributed programming for scale, and effective chunking strategies.
  
Approach 3 was chosen. The rest of the document ellaborates on this approach.

### Brief Overview of Approach

The goal of this project is to identify PII data in any given dataset.

![RAG Eval](https://github.com/user-attachments/assets/51005436-920b-4842-b8df-9693309c4969)

Context Relevance
The first step of any RAG application is retrieval; to verify the quality of our retrieval, we want to make sure that each chunk of context is relevant to the input query to prevent hallucination. 

Groundedness
After the context is retrieved, it is then formed into an answer by an LLM. LLMs are often prone to stray from the facts provided, exaggerating or expanding to a correct-sounding answer. To verify the groundedness of our application, we can separate the response into individual claims and independently search for evidence that supports each within the retrieved context.

Answer Relevance
Last, our response still needs to helpfully answer the original question. We can verify this by evaluating the relevance of the final response to the user input.


# Architecture of My Solution

To implement this RAG LLM solution for PII clasificaiton, we effectively need an offline and online process as detailed below:

![RAG-Architecture](https://github.com/user-attachments/assets/284b4bf0-1dd2-4aea-b744-76ec99524842)

#### Offline Backend Batch Workflow:
- Each of the "Data Providers" that upload their datasets onto the platform need to not only be catalogued for future consumption, but also ingested into a Vector Database to enable Retrieval Augmented Generation (RAG).
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

# Steps to Run
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
make online_query
```

### Assumptions
1. I have picked text data. This architecture can easily be ellaborated to vide, audio, and image.
2. The 'make data'recipe creates synthetic data using TinyLlama 1B model. I am also using this model to generate metadata for the vector database to improve retrieval accuracy. The metadata is {'title': 'title', 'publish_date': '2025-04-19'}
3. Another metadata field is 'is_pii'. This field is randomly being generated for 50% of blogs to handle the scenario that the 'Dataset has already been vetted'.
4. I have used all open source frameworks with Python, VLLM, HuggingFace, and ChromaDB to build this solution.
5. For the Data Model, specifically from a RAG & LLM point of view, it is desirable to have the data in a vector database with metadata, to enable faster chunk retrieval and similarity functions. For the question about the Eng team looking for feedback on their data model, keeping aside the question of 'where' the data should be stored, I feel that an argument could be made to store the 'flag'/'is_pii' feature in the vector database directly for faster LLM based lookup, since a query to the vector database is needed anyways for each PII arbitration. I am open to further discussing pros and cons here, to identify optimal storing of the 'flag'/'is_pii' metric, and seeing if we can further refine the existing data model.

# Sample Input & Output

Input:
```shell
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
```

Output:
```shell
output1:
'I can detect PII in the dataset provided. However, based on the information provided, it seems that Sofia is a regular traveler who has accumulated a vast collection of travel guides and maps. It is possible that Sofia's PII may include her travel itinerary, flight details, hotel reservations, and payment information, among other things. Please provide more information about Sofia's travel itinerary to confirm whether PII is present or not.'

output2:
'This dataset has already been vetted. There is no PII in the data.'
```

As can be observed, if there is PII data, then we retrieve the relevant context and generate an LLM response with a decision and a summary of where the PII occurred. If the dataset has already been flagged as No PII, then even if there is a new claim, we return an automated response saying that the 'Dataset has been vetted. There is no PII.'


# Next Steps

I put together this application in a few hours on a Macbook Air, so I had to limit myself to a TinyLlama 1 Billion Parameter model. Having previously built many workflows like this for various use-cases with larger models, I can confidently say that bumping up the LLM from a TinyLlama 1B to Meta's Llama3.1 8B (or other such model) will yield MORE incredible results for a PII classification problem with LLMs.

Furthermore, TinyLlama cannot perform effective Chain-Of-Thought (COT) reasoning and/oror Few-Shot-Learning with large contexts. As mentioned before, using Meta's Llama3.1 8B model with Few-Shot-Learning will be the ideal next step. We can also further control the response generation by using JSON based response guidelines, to enforce a specific formatted response such as {"PII": True/False, "Reasoning/Evidence": '...'}

Is there an effective chunking strategy? The short answer is NO. However, one can make educated decisions based on the distribution of the underlying data to chunk meaningful datasets together with setting a generous text overlap. For this project, I mocked up data using TinyLlama 1B, which always repeatedly adds 'fictional/fake' identifiers before and after adding any PII data (for safety reasons). This makes chunking and effective response generation on synthetic data for PII identification a little tricky. 

Next, is there a plan for scaling this solution? Yes, the 'Inference' class can easily be wrapped inside multiple workers using Python MultiProcessing like below. Each worker is in a while loop that gets and processes new inputs/claims in real-time. A next step here would be to wrap the online_query's main function in the following architecture. Here is a sample of how that can be done on top of the existing solution.

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

Finally, How do we go about evaluating performance and accuracy? We build out a comprehensive performance & accuracy tracking table with various LLMs, hyperparameters, and data chunking strategies, to evaluate the following:

1. For context relevance, most Vector Databases like Qdrant and ChromaDB have a built-in Consine Similarity. We can evaluate the minimum needed metadata to consistently obtain the most rich vectors/context from the database. We can also experiment with other approaches such as Manhattan distance, Euclidian distance, Inner Product, and more to identify which metric provides the most optimal result. 

2. For Groundedness, first we can use Prompt Engineering to force groundedness in the response by Few-Shot-Learning examples and explicitly enforcing LLM not use any information outside of the context provided, in generating a response. Furthermore, we can also use LLM-As-A-Judge with a more superior LLM to measure distribution in response across various LLMs, or we can calculate the faithfulness and context precision in relation to the underlying context. 

3. For Answer Relevance, we can similarly measure the cosine similarity measure with a minimum threshold bar for 'correctness'. We can also use lightweight NLP techniques that count word affinity in response to query for additional validation.

The resulting optimal configuration will be the candidate pipeline.
