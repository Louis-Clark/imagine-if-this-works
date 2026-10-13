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
    def __init__(self, n_gram=2, max_keys=5000):
        self.n_gram = n_gram
        self.max_keys = max_keys
        self.weights = {}

    def to_json(self):
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

    match = re.search(r'\[MODEL_MEMORY:\s*(\{.*?\})\s*\]', full_text)
    if match:
        full_memory_string = match.group(0)
        saved_memory = match.group(1)

    return full_text, saved_memory, full_memory_string, all_tabs

def find_dochead_occurrences_across_tabs(all_tabs):
    occurrences = []
    for tab in all_tabs:
        tab_properties = tab.get('tabProperties', {})
        tab_id = tab_properties.get('tabId')
        document_tab = tab.get('documentTab', {})
        body = document_tab.get('body', {}).get('content', [])
        
        for element in body:
            if 'paragraph' in element:
                for elem in element['paragraph'].get('elements', []):
                    if 'textRun' in elem:
                        text = elem['textRun'].get('content', '')
                        start_idx = elem.get('startIndex', 1)
                        for match in re.finditer(r'\bdochead\b', text, re.IGNORECASE):
                            match_start = start_idx + match.start()
                            match_end = start_idx + match.end()
                            occurrences.append({
                                'start': match_start,
                                'end': match_end,
                                'tabId': tab_id
                            })
    return occurrences

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
    model = DocMarkovNeuralMemory(n_gram=2, max_keys=5000)
    if saved_memory:
        model.from_json(saved_memory)
        print("Loaded neural memory from document storage block.")
    else:
        print("Initializing base neural memory weights.")
        model.train("dochead initialization core online system active protocol ready")

    # 3. Train on everything written across all tabs
    model.train(full_text)

    # 4. Find all 'dochead' occurrences across all tabs
    occurrences = find_dochead_occurrences_across_tabs(all_tabs)
    requests = []

    if occurrences:
        print(f"Found {len(occurrences)} 'dochead' trigger(s). Generating unique responses for each...")
        
        # Sort in reverse order (highest index/tab first) to prevent index shifting collisions
        occurrences.sort(key=lambda x: (str(x['tabId']), x['start']), reverse=True)
        
        for item in occurrences:
            start = item['start']
            end = item['end']
            tab_id = item['tabId']
            
            # Generate a completely FRESH, unique response for THIS specific trigger occurrence
            response = model.generate(max_words=14)
            reply_text = f"[GLITCH_BOT]: {response}"
            
            range_arg = {'startIndex': start, 'endIndex': end}
            location_arg = {'index': start}
            if tab_id:
                range_arg['tabId'] = tab_id
                location_arg['tabId'] = tab_id

            # Delete the specific 'dochead' instance and insert its unique response right in its place
            requests.append({'deleteContentRange': {'range': range_arg}})
            requests.append({'insertText': {'location': location_arg, 'text': reply_text}})
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
            # Fallback re-creation if memory tag was cleared
            first_tab = all_tabs[0]
            tab_properties = first_tab.get('tabProperties', {})
            tab_id = tab_properties.get('tabId')
            document_tab = first_tab.get('documentTab', {})
            body_content = document_tab.get('body', {}).get('content', [])
            tab_end_index = body_content[-1].get('endIndex', 1) - 1 if body_content else 1
            
            location_arg = {'index': tab_end_index}
            if tab_id:
                location_arg['tabId'] = tab_id

            requests.append({'insertText': {'location': location_arg, 'text': f"\n{reply_text}\n"}})
        
        model.train(reply_text)

    # Execute Step 1: Handle triggers or autonomous spouts
    if requests:
        service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}
        ).execute()

    # 5. Step 2: Refresh document state and update the persistent memory block
    doc_fresh = service.documents().get(documentId=document_id, includeTabsContent=True).execute()
    _, _, fresh_memory_string, _ = extract_all_docs_data(doc_fresh)
    
    new_memory_block = f"[MODEL_MEMORY: {model.to_json()}]"
    memory_requests = []
    
    target_string = fresh_memory_string if fresh_memory_string else "[MODEL_MEMORY_PENDING]"
    
    memory_requests.append({
        'replaceAllText': {
            'containsText': {'text': target_string, 'matchCase': True},
            'replaceText': new_memory_block
        }
    })

    service.documents().batchUpdate(
        documentId=document_id, body={'requests': memory_requests}
    ).execute()
    
    print("Document and neural memory updated successfully across all tabs!")

if __name__ == '__main__':
    main()
