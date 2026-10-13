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
                        if "[MODEL_MEMORY:" in content:
                            try:
                                start = content.index("[MODEL_MEMORY:") + len("[MODEL_MEMORY:")
                                end = content.index("]", start)
                                saved_memory = content[start:end].strip()
                            except ValueError:
                                pass
    return full_text, saved_memory, all_tabs

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
                                'text': text,
                                'tabId': tab_id
                            })
    return occurrences

def generate_garbled_text():
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;':\",./<>?~`~# "
    length = random.randint(20, 50)
    return "".join(random.choice(alphabet) for _ in range(length)) + "\n"

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

    # 1. Fetch document state with tabs content enabled
    doc = service.documents().get(documentId=document_id, includeTabsContent=True).execute()
    full_text, saved_memory, all_tabs = extract_all_docs_data(doc)

    # 2. Initialize and load model memory
    model = DocMarkovNeuralMemory(n_gram=2)
    if saved_memory:
        model.from_json(saved_memory)
        print("Loaded neural memory from document storage block.")
    else:
        print("Initializing base neural memory weights.")
        model.train("dochead initialization core online system active protocol ready")

    # 3. Train on everything written across all tabs
    model.train(full_text)

    # 4. Search for all 'dochead' occurrences across ALL tabs
    occurrences = find_dochead_occurrences_across_tabs(all_tabs)
    requests = []

    if occurrences:
        print(f"Found {len(occurrences)} 'dochead' trigger(s) across all document tabs.")
        occurrences.sort(key=lambda x: x['start'], reverse=True)
        
        for item in occurrences:
            start = item['start']
            end = item['end']
            context_text = item['text']
            tab_id = item['tabId']
            
            context_words = context_text.split()
            response = model.generate(seed_words=context_words, max_words=14)
            reply_text = f"\n[GLITCH_BOT]: {response}\n"
            
            range_arg = {'startIndex': start, 'endIndex': end}
            location_arg = {'index': start}
            if tab_id:
                range_arg['tabId'] = tab_id
                location_arg['tabId'] = tab_id

            requests.append({'deleteContentRange': {'range': range_arg}})
            requests.append({'insertText': {'location': location_arg, 'text': reply_text}})
    else:
        # Pick a random tab and safely append to the end of its body
        target_tab = random.choice(all_tabs)
        tab_properties = target_tab.get('tabProperties', {})
        tab_id = tab_properties.get('tabId')
        document_tab = target_tab.get('documentTab', {})
        body_content = document_tab.get('body', {}).get('content', [])
        
        tab_end_index = body_content[-1].get('endIndex', 1) - 1 if body_content else 1
        
        glitch_text = f"\n[GLITCH_BOT]: {generate_garbled_text()}\n"
        print(f"No triggers found. Injecting random glitch into tab '{tab_properties.get('title', 'Untitled')}' at index {tab_end_index}.")
        
        location_arg = {'index': tab_end_index}
        if tab_id:
            location_arg['tabId'] = tab_id

        requests.append({'insertText': {'location': location_arg, 'text': glitch_text}})
        model.train(glitch_text)

    # Execute Step 1: Content replacements / triggers
    if requests:
        service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}
        ).execute()

    # 5. Step 2: Fetch fresh document state to get the exact new end index for memory storage
    doc_fresh = service.documents().get(documentId=document_id, includeTabsContent=True).execute()
    fresh_tabs = get_all_tabs(doc_fresh)
    first_tab_fresh = fresh_tabs[0]
    first_tab_id = first_tab_fresh.get('tabProperties', {}).get('tabId')
    document_tab_fresh = first_tab_fresh.get('documentTab', {})
    body_content_fresh = document_tab_fresh.get('body', {}).get('content', [])
    doc_end_index = body_content_fresh[-1].get('endIndex', 1) - 1 if body_content_fresh else 1

    bottom_memory_block = f"\n[MODEL_MEMORY: {model.to_json()}]\n"
    memory_requests = []

    end_location_arg = {'index': doc_end_index}
    if first_tab_id:
        end_location_arg['tabId'] = first_tab_id

    memory_requests.append({
        'insertText': {
            'location': end_location_arg,
            'text': bottom_memory_block
        }
    })
    
    text_style_range = {
        'startIndex': doc_end_index,
        'endIndex': doc_end_index + len(bottom_memory_block)
    }
    if first_tab_id:
        text_style_range['tabId'] = first_tab_id

    memory_requests.append({
        'updateTextStyle': {
            'range': text_style_range,
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

    # Execute Step 2: Append & hide model memory block
    result = service.documents().batchUpdate(
        documentId=document_id, body={'requests': memory_requests}
    ).execute()
    print(f"Document updated successfully across tabs! Reply: {result}")

if __name__ == '__main__':
    main()
