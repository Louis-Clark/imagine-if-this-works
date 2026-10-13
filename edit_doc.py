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
    def __init__(self, n_gram=2, max_keys=250):
        self.n_gram = n_gram
        self.max_keys = max_keys
        self.weights = {}

    def to_json(self):
        # Keep JSON strictly single-line and compact to prevent API matching errors
        return json.dumps(self.weights, separators=(',', ':'))

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
            
        # Cap weights size to prevent exponential bloat and API precondition failures
        if len(self.weights) > self.max_keys:
            sorted_keys = sorted(
                self.weights.keys(), 
                key=lambda k: sum(self.weights[k].values()), 
                reverse=True
            )
            self.weights = {k: self.weights[k] for k in sorted_keys[:self.max_keys]}

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

def add_current_and_child_tabs(tab, all_tabs):
    all_tabs.append(tab)
    for child_tab in tab.get('childTabs', []):
        add_current_and_child_tabs(child_tab, all_tabs)

def get_all_tabs(doc):
    all_tabs = []
    for tab in doc.get('tabs', []):
        add_current_and_child_tabs(tab, all_tabs)
    return all_tabs

def extract_all_docs_data(doc):
    full_text = ""
    saved_memory = None
    full_memory_string = None
    all_tabs = get_all_tabs(doc)

    for tab in all_tabs:
        document_tab = tab.get('documentTab', {})
        body = document_tab.get('body', {}).get('content', [])
        for element in body:
            if 'paragraph' in element:
                for elem in element['paragraph'].get('elements', []):
                    if 'textRun' in elem:
                        content = elem['textRun'].get('content', '')
                        full_text += content + "\n"

    # Securely match the single-line memory block
    match = re.search(r'\[MODEL_MEMORY:\s*(\{.*?\})\s*\]', full_text)
    if match:
        full_memory_string = match.group(0)
        saved_memory = match.group(1)

    return full_text, saved_memory, full_memory_string, all_tabs

def generate_garbled_text():
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;':\",./<>?~`~# "
    length = random.randint(20, 50)
    return "".join(random.choice(alphabet) for _ in range(length))

def main():
    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    document_id = os.environ.get("GOOGLE_DOC_ID")

    if not service_account_json or not document_id:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID environment variables.")

    creds_info = service_account.Credentials.from_service_account_info(
        eval(service_account_json),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    # 1. Fetch document state across all tabs
    doc = service.documents().get(documentId=document_id, includeTabsContent=True).execute()
    full_text, saved_memory, full_memory_string, all_tabs = extract_all_docs_data(doc)

    # 2. Initialize and load model memory
    model = DocMarkovNeuralMemory(n_gram=2, max_keys=250)
    if saved_memory:
        model.from_json(saved_memory)
        print("Loaded neural memory from document storage block.")
    else:
        print("Initializing base neural memory weights.")
        model.train("dochead initialization core online system active protocol ready")

    # 3. Train on everything written across all tabs
    model.train(full_text)

    requests = []

    # 4. Check for 'dochead' triggers or fallback to autonomous spouting
    if re.search(r'\bdochead\b', full_text, re.IGNORECASE):
        print("Found 'dochead' trigger(s) across document tabs. Responding...")
        response = model.generate(max_words=14)
        reply_text = f"[GLITCH_BOT]: {response}"
        
        requests.append({
            'replaceAllText': {
                'containsText': {'text': 'dochead', 'matchCase': False},
                'replaceText': reply_text
            }
        })
    else:
        print("No 'dochead' triggers found. Spouting random glitch into document stream...")
        spout_content = model.generate(max_words=12) if len(model.weights) > 0 else generate_garbled_text()
        reply_text = f"\n[GLITCH_BOT (Autonomous Spout)]: {spout_content}\n"
        
        if full_memory_string:
            placeholder = "[MODEL_MEMORY_PENDING]"
            requests.append({
                'replaceAllText': {
                    'containsText': {'text': full_memory_string, 'matchCase': True},
                    'replaceText': f"{reply_text}\n{placeholder}"
                }
            })
            full_memory_string = placeholder
        else:
            raise ValueError("No [MODEL_MEMORY: {...}] block found in the document. Please add one at the bottom of your first tab.")
        
        model.train(reply_text)

    # 5. Update/Store the capped neural memory weights into the document block
    new_memory_block = f"[MODEL_MEMORY: {model.to_json()}]"
    if full_memory_string:
        requests.append({
            'replaceAllText': {
                'containsText': {'text': full_memory_string, 'matchCase': True},
                'replaceText': new_memory_block
            }
        })

    # Execute batch update cleanly with zero index risk
    if requests:
        result = service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}
        ).execute()
        print(f"Document updated successfully across tabs! Result: {result}")
    else:
        print("No updates required.")

if __name__ == '__main__':
    main()
