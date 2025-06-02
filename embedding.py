from openai import OpenAI
import numpy as np

# Initialize the OpenAI client
client = OpenAI()


# Function to get the embedding vector of a given text
def get_embedding(text: str) -> np.ndarray:
    # Request an embedding from the OpenAI API using the specified model
    response = client.embeddings.create(input=text, model="text-embedding-3-small")

    # Extract the embedding vector from the response
    res = response.data[0].embedding

    # Convert the embedding list to a NumPy array and return it
    return np.array(res)
