import time
import logging
import functools
import numpy as np
import faiss
import torch
import torch.nn as nn
from openai import OpenAI
from app.sqlite3db import SQLPostingDatabase

faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)

client = OpenAI()

# Initialize the database driver
driver = SQLPostingDatabase()

# Load the refined embeddings
reduced_embeddings = np.load("data/reduced_embeddings.npy").astype("float32")


# Load job data (titles and descriptions)
def get_jobs_data():
    jobs = driver.get_all()
    payload = {
        "job_id": [],
        "job_title": [],
        "job_description": [],
        "company": [],
        "location": [],
        "url": [],
    }
    for job in jobs:
        payload["job_id"].append(job["id"])
        payload["job_title"].append(job["title"])
        payload["job_description"].append(job["description"])
        payload["company"].append(job["company"])
        payload["location"].append(job["location"])
        payload["url"].append(job["url"])
    return payload


data = get_jobs_data()

# Define the DenoisingAutoencoder class
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size=3072):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Initialize the model and load weights
logger.info("Loading model...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info("Using device: %s", device)
model = DenoisingAutoencoder(input_size=3072).to(device)
loaded_model = torch.load("data/denoising_autoencoder.pth", map_location=device)
logger.info("Model loaded successfully.")
model.load_state_dict(loaded_model)
logger.info("Model state dict loaded successfully.")
model.eval()
logger.info("Model ready for inference.")


# Precompute normalized job embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


job_embeddings_norm = normalize_embeddings(reduced_embeddings)

# Build FAISS index for similarity search
d = reduced_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(job_embeddings_norm)


@functools.lru_cache(maxsize=128)
def generate_query_embedding(query_text):
    try:
        response = client.embeddings.create(
            input=[query_text], model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
        query_embedding = np.array(query_embedding)
        return query_embedding
    except Exception as e:
        logger.error("Error generating query embedding: %s", e)
        return [0] * 3072


def get_possible_jobs(resume_text, personal_interests):
    prompt = (
        f"Considerando os interesses pessoais: '{personal_interests}', "
        "quais carreiras seriam mais adequadas para essa pessoa baseado nos interesses pessoais?"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
        )
        llm_response = response.choices[0].message.content
        print(llm_response)
        return llm_response
    except Exception as e:
        logger.error("Error generating career recommendations: %s", e)
        return "Não foi possível gerar recomendações de carreiras no momento."


def get_user_profile_embedding(resume_text, personal_interests, alpha=0.6, beta=0.4):
    """
    Gera o embedding do perfil do usuário combinando o currículo e os interesses pessoais.
    Os pesos alpha e beta determinam a influência de cada componente.
    """
    # Gerar embeddings separados
    resume_embedding = generate_query_embedding(resume_text)

    # Obter carreiras possíveis com base no currículo e interesses pessoais
    possible_jobs = get_possible_jobs(resume_text, personal_interests)

    # Gerar embedding para carreiras possíveis
    interests_embedding = generate_query_embedding(possible_jobs)

    # Combinar embeddings com pesos
    user_embedding = alpha * np.array(resume_embedding) + beta * np.array(
        interests_embedding
    )
    return user_embedding

def get_recommendations(user_embedding, threshold=0.33, top_n=3500, max_jobs_per_company=5, min_similarity_diff=0.005):
    """
    Adiciona um filtro para evitar jobs muito semelhantes dentro da mesma empresa.
    """
    start_time = time.time()

    # Convert to torch tensor
    user_embedding_tensor = torch.FloatTensor(user_embedding).to(device)

    # Encode the user embedding using the encoder
    with torch.no_grad():
        user_embedding_tensor = user_embedding_tensor.unsqueeze(0)  # Add batch dimension
        user_encoded = model.encoder(user_embedding_tensor).cpu().numpy()

    # Normalize the user encoded vector
    user_encoded_norm = user_encoded / np.linalg.norm(user_encoded)
    user_encoded_norm = user_encoded_norm.astype("float32")

    # Compute cosine similarities using FAISS index
    k = min(top_n * 5, job_embeddings_norm.shape[0])  # Retrieve more than needed
    D, I = index.search(user_encoded_norm, k)

    # D is similarities, I is indices
    similarities = D.flatten()
    indices = I.flatten()

    # Filter out similarities below threshold
    valid = similarities >= threshold
    similarities = similarities[valid]
    indices = indices[valid]

    if len(indices) == 0:
        return []

    results = []
    company_count = {}  # Track the number of jobs added per company
    selected_similarities = {}  # Track last similarity for each company

    for idx, sim in zip(indices, similarities):
        company = data["company"][idx]

        # Skip if company exceeds max allowed jobs
        if company_count.get(company, 0) >= max_jobs_per_company:
            continue

        # Skip if the similarity is too close to the previous ones for this company
        if company in selected_similarities and abs(sim - selected_similarities[company]) < min_similarity_diff:
            continue

        job = {
            "job_id": data["job_id"][idx],
            "job_title": data["job_title"][idx],
            "job_description": data["job_description"][idx],
            "company": company,
            "location": data["location"][idx],
            "url": data["url"][idx],
            "similarity": float(sim),
        }
        results.append(job)

        # Update company count and similarity
        company_count[company] = company_count.get(company, 0) + 1
        selected_similarities[company] = sim

        if len(results) >= top_n:
            break

    end_time = time.time()
    logger.info("Query processed in %.2f seconds.", end_time - start_time)

    return {
        "time_taken": end_time - start_time,
        "message": "OK",
        "results": results,
    }
