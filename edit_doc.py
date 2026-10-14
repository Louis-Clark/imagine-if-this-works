import os
import random
import json
import re
import math
import string
from collections import Counter
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")


class DocMarkovNeuralMemory:
    def __init__(self, n_gram=2, max_keys=250, max_vocab=200):
        self.n_gram = n_gram
        self.max_keys = max_keys
        self.max_vocab = max_vocab
        self.weights = {}          # markov table
        self.vocab = {}            # token -> {count, book_count, first_tick}
        self.mood = 0.5            # 0.0 cold/coherent, 1.0 hot/glitchy
        self.tick = 0

    # ---------- serialization ----------
    def to_json(self):
        payload = {
            "w": self.weights,
            "v": self.vocab,
            "m": round(self.mood, 4),
            "t": self.tick,
        }
        return json.dumps(payload, separators=(',', ':'))

    def from_json(self, json_str):
        try:
            payload = json.loads(json_str)
            # backwards-compat: old format was just the weights dict
            if "w" in payload and isinstance(payload.get("w"), dict):
                self.weights = payload.get("w", {})
                self.vocab = payload.get("v", {})
                self.mood = float(payload.get("m", 0.5))
                self.tick = int(payload.get("t", 0))
            else:
                self.weights = payload
                self.vocab = {}
                self.mood = 0.5
                self.tick = 0
        except Exception:
            self.weights = {}
            self.vocab = {}
            self.mood = 0.5
            self.tick = 0

    # ---------- training ----------
    def train(self, text, is_self=False):
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
            # self-generated text counts a bit less than book text, to slow inbreeding
            self.weights[key][next_word] = current_score + (0.6 if is_self else 1.0)

        # track vocab for naming
        for w in words:
            clean = w.strip(string.punctuation).lower()
            if len(clean) < 4:
                continue
            entry = self.vocab.setdefault(clean, {"c": 0, "b": 0, "f": self.tick})
            entry["c"] += 1
            if not is_self:
                entry["b"] += 1

        # cap weights
        if len(self.weights) > self.max_keys:
            sorted_keys = sorted(
                self.weights.keys(),
                key=lambda k: sum(self.weights[k].values()),
                reverse=True
            )
            self.weights = {k: self.weights[k] for k in sorted_keys[:self.max_keys]}

        # cap vocab, but never prune [NEW] names
        if len(self.vocab) > self.max_vocab:
            named = {k: v for k, v in self.vocab.items() if k.startswith("[new]")}
            others = {k: v for k, v in self.vocab.items() if not k.startswith("[new]")}
            top_others = dict(sorted(others.items(), key=lambda kv: kv[1]["c"], reverse=True)[:self.max_vocab - len(named)])
            self.vocab = {**named, **top_others}

    # ---------- naming ----------
    def find_nameable_tokens(self, min_self=3, ratio=2.5):
        """Find tokens the organism says a lot but the book doesn't."""
        candidates = []
        for tok, meta in self.vocab.items():
            if tok.startswith("[new]"):
                continue
            if len(tok) < 5:
                continue
            self_c = meta["c"] - meta["b"]
            if self_c < min_self:
                continue
            book_c = meta["b"] + 1
            if self_c / book_c >= ratio:
                candidates.append((tok, self_c, book_c))
        candidates.sort(key=lambda x: x[1] / x[2], reverse=True)
        return candidates[:3]

    def promote_to_name(self, token):
        """Mark a token as belonging to the organism."""
        entry = self.vocab.get(token)
        if not entry or entry.get("named"):
            return False
        entry["named"] = True
        entry["f"] = self.tick
        # also inject as its own key so generation can produce it whole
        named_form = f"[NEW]{token}"
        self.weights.setdefault(named_form, {})[named_form] = 1.0
        return True

    # ---------- mood ----------
    def update_mood(self, last_output, had_trigger):
        """Homeostatic drift based on repetition, loneliness, and novelty."""
        if last_output:
            toks = last_output.split()
            if len(toks) > 1:
                uniq = len(set(toks)) / len(toks)
                # low diversity -> mood rises (escape the attractor)
                self.mood += (0.5 - uniq) * 0.15
            # novelty from [NEW] tokens cools the mood
            new_ratio = sum(1 for t in toks if t.startswith("[NEW]")) / max(len(toks), 1)
            self.mood -= new_ratio * 0.1

        if not had_trigger:
            # loneliness raises mood over time
            self.mood += 0.02

        self.mood = max(0.05, min(0.95, self.mood))

    # ---------- generation ----------
    def generate(self, seed_words=None, max_words=14):
        if not self.weights:
            return "system_matrix_tabula_rasa_online..."

        temperature = 0.5 + self.mood * 1.2  # 0.5 .. ~1.6

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

            # temperature-scaled softmax
            logits = [math.log(max(s, 1e-9)) / max(temperature, 0.1) for s in scores]
            m = max(logits)
            exps = [math.exp(l - m) for l in logits]
            total = sum(exps)
            probs = [e / total for e in exps]

            chosen = random.choices(choices, weights=probs, k=1)[0]
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
    model = DocMarkovNeuralMemory(n_gram=2, max_keys=250, max_vocab=200)
    if saved_memory:
        model.from_json(saved_memory)
        print(f"Loaded neural memory. mood={model.mood:.2f} tick={model.tick}")
    else:
        print("Initializing base neural memory weights.")
        model.train("dochead initialization core online system active protocol ready")

    # 3. Train on everything written across all tabs (this is the book + history)
    model.train(full_text, is_self=False)

    requests = []
    had_trigger = False
    last_output = ""

    # 4. Trigger or spout
    if re.search(r'\bdochead\b', full_text, re.IGNORECASE):
        had_trigger = True
        print("Found 'dochead' trigger(s) across document tabs. Responding...")
        response = model.generate(max_words=14)
        last_output = response
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
        last_output = spout_content
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

    # 5a. Feed its own output back in (self-training, weighted lower)
    model.train(reply_text, is_self=True)

    # 5b. Naming: promote tokens the organism keeps saying that the book doesn't
    named_this_tick = []
    for tok, self_c, book_c in model.find_nameable_tokens():
        if model.promote_to_name(tok):
            named_this_tick.append(tok)
            # annotate in-document so humans can see the birth of a word
            requests.append({
                'replaceAllText': {
                    'containsText': {'text': tok, 'matchCase': False},
                    'replaceText': f"[NEW]{tok}"
                }
            })

    # 5c. Mood update
    model.update_mood(last_output, had_trigger)
    model.tick += 1

    # 6. Persist memory (weights + vocab + mood + tick)
    new_memory_block = f"[MODEL_MEMORY: {model.to_json()}]"
    if full_memory_string:
        requests.append({
            'replaceAllText': {
                'containsText': {'text': full_memory_string, 'matchCase': True},
                'replaceText': new_memory_block
            }
        })

    # 7. Execute batch
    if requests:
        result = service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}
        ).execute()
        print(f"Document updated. mood={model.mood:.2f} tick={model.tick} named={named_this_tick}")
    else:
        print("No updates required.")


if __name__ == '__main__':
    main()
