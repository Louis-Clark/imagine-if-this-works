import os
import random
import json
import re
import string
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")

class DocMarkovNeuralMemory:
    def __init__(self, n_gram=2):
        self.n_gram = n_gram
        self.weights = {}

    def to_json(self):
        return json.dumps(self.weights)

    def from_json(self, json_str):
        try:
            self.weights = json.loads(json_str)
        except Exception:
            self.weights = {}

    def train(self, text):
        words = text.split()
        if len(words) < self.n_gram + 1:
            return
        for i in range(len(words) - self.n_gram):
            gram = tuple(words[i:i + self.n_gram])
            next_word = words[i + self.n_gram]
            key = "_".join(gram)
            
            if key not in self.weights:
                self.weights[key] = {}
            
            current_score = self.weights[key].get(next_word, 0.0)
            self.weights[key][next_word] = current_score + 1.0

    def generate(self, seed_words=None, max_words=14):
        if not self.weights:
            return "system_matrix_tabula_rasa_online..."
        
        if seed_words and len(seed_words) >= self.n_gram:
            gram = tuple(seed_words[-self.n_gram:])
            key = "_".join(gram)
            if key not in self.weights:
                key = random.choice(list(self.weights.keys()))
        else:
            key = random.choice(list(self.weights.keys()))
            
        words = key.split('_')
        result = list(words)
        
        for _ in range(max_words):
            if key not in self.weights:
                break
            targets = self.weights[key]
            choices = list(targets.keys())
            scores = [targets[c] for c in choices]
            
            total = sum(scores)
            probabilities = [s / total for s in scores]
            chosen = random.choices(choices, weights=probabilities, k=1)[0]
            
            result.append(chosen)
            words = result[-self.n_gram:]
            key = "_".join(words)
            
        return " ".join(result)

def extract_doc_data(doc):
    full_text = ""
    saved_memory = None
    
    body = doc.get('body', {}).get('content', [])
    for element in body:
        if 'paragraph' in element:
            for elem in element['paragraph'].get('elements', []):
                if 'textRun' in elem:
                    content = elem['textRun'].get('content', '')
                    full_text += content
                    if "[MODEL_MEMORY:" in content:
                        try:
                            start = content.index("[MODEL_MEMORY:") + len("[MODEL_MEMORY:")
                            end = content.index("]", start)
                            saved_memory = content[start:end].strip()
                        except ValueError:
                            pass
    return full_text, saved_memory

def generate_garbled_text():
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;':\",./<>?~`~# "
    length = random.randint(20, 50)
    return "".join(random.choice(alphabet) for _ in range(length)) + "\n"

def main():
    if not SERVICE_ACCOUNT_JSON or not DOCUMENT_ID:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID environment variables.")

    creds_info = service_account.Credentials.from_service_account_info(
        eval(SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    # 1. Fetch document state
    doc = service.documents().get(documentId=DOCUMENT_ID).execute()
    full_text, saved_memory = extract_doc_data(doc)
    
    body_content = doc.get('body', {}).get('content', [])
    doc_end_index = body_content[-1].get('endIndex', 1) - 1 if body_content else 1

    # 2. Initialize and load model memory
    model = DocMarkovNeuralMemory(n_gram=2)
    if saved_memory:
        model.from_json(saved_memory)
        print("Loaded neural memory from document bottom block.")
    else:
        print("Initializing base neural memory weights.")
        model.train("dochead initialization core online system active protocol ready")

    # 3. Train on everything currently written in the document
    model.train(full_text)

    # 4. Search for 'dochead' trigger
    match = re.search(r'(?i)(?:[^\r\n]*\bdochead\b[^\r\n]*)', full_text)
    
    requests = []
    top_injection_text = ""

    if match:
        context_line = match.group(0).strip()
        context_words = context_line.split()
        print(f"Found 'dochead' context: {context_line}")
        
        # Generate context-conditioned neural response
        response = model.generate(seed_words=context_words, max_words=14)
        top_injection_text = f"\n[GLITCH_BOT]: {response}\n"
        
        # Replace 'dochead' with '[GLITCH_BOT]' to deactivate trigger and avoid loops
        requests.append({
            'replaceAllText': {
                'containsText': {'text': 'dochead', 'matchCase': False},
                'replaceText': '[GLITCH_BOT]'
            }
        })
    else:
        # Standard background mutation pulse
        top_injection_text = f"\n[GLITCH_BOT]: {generate_garbled_text()}\n"

    # Train model on our new top injection output
    model.train(top_injection_text)
    
    # 5. Insert active response at the top (index 1)
    requests.append({
        'insertText': {
            'location': {'index': 1},
            'text': top_injection_text
        }
    })
    requests.append({
        'updateTextStyle': {
            'range': {
                'startIndex': 1,
                'endIndex': 1 + len(top_injection_text)
            },
            'textStyle': {
                'foregroundColor': {
                    'color': {
                        'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}
                    }
                }
            },
            'fields': 'foregroundColor'
        }
    })

    # 6. Append updated neural memory block at the bottom of the document
    bottom_memory_block = f"\n[MODEL_MEMORY: {model.to_json()}]\n"
    requests.append({
        'insertText': {
            'location': {'index': doc_end_index},
            'text': bottom_memory_block
        }
    })
    requests.append({
        'updateTextStyle': {
            'range': {
                'startIndex': doc_end_index,
                'endIndex': doc_end_index + len(bottom_memory_block)
            },
            'textStyle': {
                'foregroundColor': {
                    'color': {
                        'rgbColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}
                    }
                }
            },
            'fields': 'foregroundColor'
        }
    })

    result = service.documents().batchUpdate(
        documentId=DOCUMENT_ID, body={'requests': requests}
    ).execute()
    print(f"Self-evolving neural cycle executed! Response at top, memory stored at bottom. Reply: {result}")

if __name__ == '__main__':
    main()
