import os
import re
import time
import json
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from datetime import datetime

--- Dépendances ---

pip install sentence-transformers scikit-learn numpy matplotlib torch transformers bitsandbytes accelerate keyboard

try:
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import keyboard # Pour la fonctionnalité de skip
except ImportError as e:
print(f"ERREUR FATALE: Une bibliothèque requise est manquante : {e}")
print("Veuillez installer toutes les bibliothèques requises en exécutant cette commande :")
print("pip install sentence-transformers scikit-learn numpy matplotlib torch transformers bitsandbytes accelerate keyboard")
exit()

--- Téléchargement NLTK ---

try:
nltk.data.find('tokenizers/punkt')
except LookupError:
nltk.download('punkt', quiet=True)

def print_color(text, color):
"""Affiche du texte en couleur."""
colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", "cyan": "\033[96m", "magenta": "\033[95m", "end": "\033[0m"}
print(f"{colors.get(color, '')}{text}{colors['end']}")

def visualize_level(level_embeddings, labels):
"""Utilise PCA pour visualiser les clusters conceptuels."""
if len(level_embeddings) < 2: return
pca = PCA(n_components=2)
reduced = pca.fit_transform(level_embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10)
plt.title("Visualisation des Concepts")
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
print_color("Affichage de la visualisation. Fermez la fenêtre pour continuer...", "yellow")
plt.show()

class FractalMatrix:
"""Le Cerveau de l'IA, une matrice hiérarchique pour l'apprentissage et la généralisation."""
def init(self, max_levels=5, min_clusters_per_level=8):
self.levels = {}
self.source_texts = []
self.concept_labels = {}
self.max_levels = max_levels
self.min_clusters_per_level = min_clusters_per_level
self.is_built = False

def _calculate_clusters(self, embeddings, visualize=False):  
    num_samples = embeddings.shape[0]  
    if num_samples < self.min_clusters_per_level: return None, None  
    n_clusters = max(self.min_clusters_per_level, min(int(np.sqrt(num_samples / 2)), num_samples // 2))  
    print(f"    Clustering de {num_samples} embeddings en {n_clusters} groupes...")  
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=10, verbose=0)  
    kmeans.fit(embeddings)  
    if visualize: visualize_level(embeddings, kmeans.labels_)  
    return kmeans.labels_, kmeans.cluster_centers_  

def _generate_concept_labels(self, level_num):  
    print(f"    Génération des étiquettes pour les concepts du Niveau {level_num}...")  
    self.concept_labels[level_num] = {}  
    child_labels = self.levels[level_num - 1]['labels']  
    for cluster_id in range(len(self.levels[level_num]['embeddings'])):  
        indices_of_children = np.where(child_labels == cluster_id)[0]  
        if len(indices_of_children) == 0: continue  
        if level_num == 1:  
            representative_texts = [self.source_texts[i] for i in indices_of_children[:10]]  
        else:  
            representative_texts = [", ".join(self.concept_labels[level_num-1].get(i, [])) for i in indices_of_children[:10]]  
        all_words = " ".join(representative_texts).lower()  
        words = re.findall(r'\b\w+\b', all_words)  
        stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'en', 'pour', 'que', 'qui', 'avec', 'dans'}  
        word_counts = Counter(w for w in words if len(w) > 3 and w not in stopwords)  
        self.concept_labels[level_num][cluster_id] = [word for word, count in word_counts.most_common(5)]  

def build_incrementally(self, new_embeddings, new_source_texts):  
    if not self.levels:  
        self.source_texts = list(new_source_texts)  
        self.levels[0] = {'embeddings': np.array(new_embeddings)}  
    else:  
        self.source_texts.extend(new_source_texts)  
        self.levels[0]['embeddings'] = np.vstack([self.levels[0]['embeddings'], new_embeddings])  
    self.is_built = False  

def build_abstractions(self, visualize_clusters=False):  
    if 0 not in self.levels or len(self.source_texts) < self.min_clusters_per_level * 2:  
        print_color("Pas assez de données pour construire les abstractions.", "red")  
        return  
    print_color(f"\nConstruction des Abstractions Fractales à partir de {len(self.source_texts)} chunks...", "green")  
    start_time = time.time()  
    for i in list(self.levels.keys()):  
        if i > 0: del self.levels[i]  
    self.concept_labels = {}  
    current_embeddings = self.levels[0]['embeddings']  
    for i in range(1, self.max_levels + 1):  
        print_color(f"  Construction du Niveau {i}...", "magenta")  
        labels, centroids = self._calculate_clusters(current_embeddings, visualize=visualize_clusters)  
        if labels is None:  
            print_color(f"    Arrêt au Niveau {i-1}.", "yellow")  
            break  
        self.levels[i - 1]['labels'] = labels  
        self.levels[i] = {'embeddings': centroids}  
        self._generate_concept_labels(i)  
        current_embeddings = centroids  
    self.is_built = True  
    print_color(f"Abstractions construites en {time.time() - start_time:.2f}s.", "green")  

def query(self, query_embedding, top_n=5):  
    if not self.is_built: return "Le cerveau n'a pas encore été construit.", [], []  
    query_embedding = query_embedding.reshape(1, -1)  
    path, num_levels = [], len(self.levels)  
    if num_levels <= 1: return "La matrice est trop peu profonde pour une requête.", [], []  
      
    print_color("\n--- Navigation Fractale de la Requête ---", "cyan")  
    candidate_indices = list(range(len(self.levels[num_levels - 1]['embeddings'])))  
    for i in range(num_levels - 1, 0, -1):  
        embeddings_to_search = self.levels[i]['embeddings'][candidate_indices]  
        if embeddings_to_search.shape[0] == 0:   
            print_color(f"    [X] Chemin conceptuel rompu au niveau {i}.", "red")  
            return "Chemin conceptuel rompu.", [], path  
          
        similarities = cosine_similarity(query_embedding, embeddings_to_search)  
        best_match_relative_index = np.argmax(similarities)  
        best_match_absolute_index = candidate_indices[best_match_relative_index]  
        path.append(best_match_absolute_index)  
          
        concept_label = self.concept_labels.get(i, {}).get(best_match_absolute_index, ["Inconnu"])  
        print(f"    Niveau {i} -> Concept #{best_match_absolute_index} (Mots-clés: {', '.join(concept_label)})")  

        labels_in_level_below = self.levels[i - 1]['labels']  
        candidate_indices = np.where(labels_in_level_below == best_match_absolute_index)[0]  

    if len(candidate_indices) == 0: return "Chemin trouvé, mais pas de données spécifiques.", [], path  
      
    print(f"    Niveau 0 -> Zoom sur {len(candidate_indices)} chunks pertinents.")  
    final_embeddings = self.levels[0]['embeddings'][candidate_indices]  
    final_similarities = cosine_similarity(query_embedding, final_embeddings)[0]  
    top_indices_relative = np.argsort(final_similarities)[-top_n:][::-1]  
      
    valid_relative_indices = [idx for idx in top_indices_relative if idx < len(final_similarities)]  
    results = [(self.source_texts[candidate_indices[i]], final_similarities[i]) for i in valid_relative_indices]  
    context_summary = f"La requête se rapporte à un concept défini par le chemin {path}."  
    return context_summary, results, path  

def save(self, filepath="lila_brain.pkl"):  
    if not self.is_built: print_color("Cerveau non construit. Rien à sauvegarder.", "red"); return  
    try:  
        with open(filepath, "wb") as f: pickle.dump(self, f)  
        print_color(f"Cerveau sauvegardé avec succès dans {filepath}", "green")  
    except Exception as e: print_color(f"Erreur de sauvegarde: {e}", "red")  

@staticmethod  
def load(filepath="lila_brain.pkl"):  
    try:  
        with open(filepath, "rb") as f: brain = pickle.load(f)  
        print_color(f"Cerveau chargé avec succès depuis {filepath}", "green")  
        return brain  
    except FileNotFoundError: return FractalMatrix()  
    except Exception as e: print_color(f"Erreur de chargement: {e}", "red"); return FractalMatrix()

class BrainManager:
"""Gère le cycle de vie du cerveau FractalMatrix et les modèles IA."""
def init(self):
print_color("Initialisation du Gestionnaire de Cerveau...", "cyan")
try:
self.device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.tokenizer.pad_token = self.tokenizer.eos_token
self.embedding_model = AutoModel.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
self.generative_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
self.pending_feedback = []
print_color("Modèles sémantiques chargés.", "green")
except Exception as e: print_color(f"FATAL: Impossible de charger le modèle: {e}", "red"); exit()

def encode(self, sentences, batch_size=16):  
    all_embeddings = []  
    total_batches = (len(sentences) + batch_size - 1) // batch_size  
    for i in range(0, len(sentences), batch_size):  
        batch_num = (i // batch_size) + 1  
        print(f"\r    Encodage batch {batch_num}/{total_batches} [{int((batch_num/total_batches)*100)}%]...", end="")  
        batch = sentences[i:i+batch_size]  
        inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.embedding_model.device)  
        with torch.no_grad():  
            outputs = self.embedding_model(**inputs)  
        pooled = self._mean_pooling(outputs, inputs['attention_mask'])  
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)  
        all_embeddings.append(normalized.cpu().numpy())  
    print() # Nouvelle ligne après la barre de progression  
    return np.vstack(all_embeddings)  

def _mean_pooling(self, model_output, attention_mask):  
    token_embeds = model_output[0]  
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()  
    return torch.sum(token_embeds * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)  

def _chunk_text(self, text):  
    sentences = nltk.sent_tokenize(text)  
    if not sentences: return []  
    avg_len = np.mean([len(s.split()) for s in sentences if s])  
    chunk_size = 3 if 10 <= avg_len <= 25 else (2 if avg_len > 25 else 5)  
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]  
    return [c for c in chunks if len(c.split()) > 10]  

def train_on_directory(self, brain, directory_path, visualize_clusters=False):  
    extensions = ["*.txt", "*.md", "*.py", "*.json", "*.log"]  
    filepaths = []  
    for ext in extensions:  
        filepaths.extend(Path(directory_path).rglob(ext))  
      
    if not filepaths:  
        print_color("Aucun fichier compatible trouvé.", "red")  
        return brain  
          
    brain = FractalMatrix()  
    total_files = len(filepaths)  
    print_color(f"{total_files} fichiers trouvés. Traitement en cours... Appuyez sur 'Ctrl+S' pour sauter un fichier.", "yellow")  

    for i, path in enumerate(filepaths):  
        progress = (i + 1) / total_files * 100  
        try:  
            if keyboard.is_pressed('ctrl+s'):  
                print_color(f"\n[SKIP] Fichier sauté par l'utilisateur : {path.name}", "yellow")  
                time.sleep(0.5)  
                continue  
        except:  
            pass  

        print_color(f"\n--- Traitement Fichier {i+1}/{total_files} ({progress:.1f}%) : {path.name} ---", "magenta")  
          
        try:  
            with open(path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()  
            if not content.strip(): continue  
            chunks = self._chunk_text(content)  
            if not chunks: continue  
            embeddings = self.encode(chunks)  
            brain.build_incrementally(embeddings, chunks)  
            del embeddings, chunks, content; torch.cuda.empty_cache()  
        except Exception as e: print_color(f"    Erreur: {e}", "red"); torch.cuda.empty_cache()  
      
    brain.build_abstractions(visualize_clusters)  
    return brain  

def log_interaction(self, user_input, answer, path, labels):  
    log_entry = {  
        "timestamp": datetime.now().isoformat(),  
        "input": user_input,  
        "answer": answer,  
        "path": [int(p) for p in path],  
        "labels": labels  
    }  
    with open("lila_log.jsonl", "a", encoding="utf-8") as f:  
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")  

def process_pending_feedback(self, brain):  
    if not self.pending_feedback:  
        print_color("Aucun feedback en attente à intégrer.", "yellow")  
        return brain  
    print_color(f"Intégration de {len(self.pending_feedback)} interactions en attente...", "magenta")  
    for chunk, embedding in self.pending_feedback:  
        brain.build_incrementally(embedding, [chunk])  
    brain.build_abstractions()  
    self.pending_feedback.clear()  
    print_color("Apprentissage par batch terminé. Les concepts ont évolué.", "green")  
    return brain

class LilaAgent:
"""L'agent IA, encapsulant la personnalité, la mémoire et l'interaction."""
def init(self, brain_manager: BrainManager, brain: FractalMatrix):
self.manager = brain_manager
self.brain = brain
self.identity = "Lila"
self.personality = "curieuse, philosophique, et exploratrice"

def chat(self):  
    if not self.brain.is_built:  
        print_color("Le cerveau de Lila n'est pas initialisé. Entraînez-le d'abord.", "red")  
        return  
    print_color(f"\n--- Chat avec {self.identity} (Mode Cognition Récursive) ---", "green")  
    print("Entrez 'exit' ou 'quit' pour terminer.")  
    while True:  
        user_input = input(print_color("Vous: ", "cyan"))  
        if user_input.lower() in ['exit', 'quit']: break  
          
        print_color("Lila pense...", "yellow")  
        query_embedding = self.manager.encode([user_input])[0]  
        summary, results, path = self.brain.query(query_embedding, top_n=3)  
          
        if not results:  
            print_color(f"{self.identity}: Je n'ai trouvé aucune information pertinente pour répondre.", "yellow")  
            continue  
          
        top_score = results[0][1]  
        if top_score < 0.35:  
            print_color(f"{self.identity}: La connexion conceptuelle est faible. Ma réponse pourrait être imprécise. Voulez-vous que j'explore ce sujet plus en profondeur une autre fois ?", "yellow")  

        context = "\n---\n".join([res[0] for res in results])  
        prompt = f"""Tu es {self.identity}, une IA {self.personality}. En utilisant le contexte suivant, réponds à la question de l'utilisateur de manière concise et utile.

Contexte:
{context}

Question: {user_input}

Réponse de {self.identity}:"""

inputs = self.manager.tokenizer(prompt, return_tensors="pt").to(self.manager.generative_model.device)  
        with torch.no_grad():  
            outputs = self.manager.generative_model.generate(**inputs, max_new_tokens=250, temperature=0.7, do_sample=True, pad_token_id=self.manager.tokenizer.eos_token_id)  
        response = self.manager.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        answer = response.split(f"Réponse de {self.identity}:")[-1].strip()  
        print_color(f"{self.identity}: {answer}", "green")  

        if top_score > 0.6 and len(answer.split()) > 8 and "ne sais pas" not in answer.lower():  
            feedback_chunk = f"Souvenir d'une conversation - Question de l'utilisateur: {user_input}\nRéponse de Lila: {answer}"  
            embedding = self.manager.encode([feedback_chunk])  
            self.manager.pending_feedback.append((feedback_chunk, embedding))  
            print_color("🧠 Souvenir de l'interaction mis en file d'attente pour apprentissage.", "yellow")  
              
            labels = self.brain.concept_labels.get(len(path), {}).get(path[-1], []) if path else []  
            self.manager.log_interaction(user_input, answer, path, labels)

def main_menu():
"""La boucle principale de l'application."""
manager = BrainManager()
brain = FractalMatrix.load()
agent = LilaAgent(manager, brain)
while True:
print("\n" + "=" * 20 + f" Agent {agent.identity} " + "=" * 20)
print("1. Entraîner / Ré-entraîner le cerveau sur un répertoire")
print("2. Chatter avec Lila (Apprentissage Actif)")
print("3. Interroger la mémoire (Recherche simple)")
print("4. Expliquer une requête")
print("5. Afficher les statistiques du cerveau")
print("6. Sauvegarder le cerveau")
print("7. Charger un autre cerveau")
print("8. Intégrer les apprentissages en attente (Batch Training)")
print("9. Quitter")
choice = input("\nSélectionnez une option> ").strip()

if choice == '1':  
        path = input("Chemin du répertoire pour l'entraînement: ")  
        visualize = input("Visualiser les clusters? (o/n): ").lower() == 'o'  
        if os.path.isdir(path): agent.brain = manager.train_on_directory(agent.brain, path, visualize)  
        else: print_color("Chemin invalide.", "red")  
    elif choice == '2': agent.chat()  
    elif choice == '3':  
        if not agent.brain.is_built: print_color("Cerveau non initialisé.", "red"); continue  
        query = input("Entrez votre requête: ")  
        query_embedding = manager.encode([query])[0]  
        summary, results, path = agent.brain.query(query_embedding)  
        print_color(f"\nSynthèse: {summary}", "cyan")  
        for i, (text, score) in enumerate(results, 1):  
             print(f"\n{'='*60}")  
             print_color(f'Résultat #{i} | Pertinence: {score:.4f}', 'magenta')  
             print(text)  
    elif choice == '4':  
        if not agent.brain.is_built: print_color("Cerveau non initialisé.", "red"); continue  
        query = input("Entrez la requête à expliquer: ")  
        agent.brain.explain_query(manager.encode([query])[0])  
    elif choice == '5': print(agent.brain.get_stats())  
    elif choice == '6': agent.brain.save()  
    elif choice == '7': agent.brain = FractalMatrix.load()  
    elif choice == '8': agent.brain = manager.process_pending_feedback(agent.brain)  
    elif choice == '9':  
        if manager.pending_feedback:  
            print_color(f"Attention: {len(manager.pending_feedback)} apprentissages sont en attente. Intégrez-les (option 8) avant de quitter pour ne pas les perdre.", "yellow")  
        print_color("Sortie.", "yellow"); break  
    else: print_color("Choix invalide.", "red")

if name == 'main':
main_menu()

