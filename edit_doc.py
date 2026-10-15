import os
import random
import json
import re
import math
import string
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")


class DocMarkovNeuralMemory:
    def __init__(self, n_gram=2, max_keys=3000, max_vocab=400, max_phrases=300):
        self.n_gram = n_gram              # primary bigram size (kept for compat)
        self.max_keys = max_keys
        self.max_vocab = max_vocab
        self.max_phrases = max_phrases

        # Bigram table (primary)
        self.weights = {}
        # Trigram table (used when context is deep enough)
        self.tri = {}

        # Phrase memory: recurring multi-word units the organism produces
        self.phrases = {}   # phrase_string -> {c: count, self_c: count}

        # Token vocab for naming
        self.vocab = {}

        self.mood = 0.5
        self.tick = 0

    # ---------- serialization ----------
    def to_json(self):
        payload = {
            "w": self.weights,
            "t3": self.tri,
            "p": self.phrases,
            "v": self.vocab,
            "m": round(self.mood, 4),
            "tk": self.tick,
        }
        return json.dumps(payload, separators=(',', ':'))

    def from_json(self, json_str):
        try:
            payload = json.loads(json_str)
            if "w" in payload and isinstance(payload.get("w"), dict):
                self.weights = payload.get("w", {})
                self.tri = payload.get("t3", {})
                self.phrases = payload.get("p", {})
                self.vocab = payload.get("v", {})
                self.mood = float(payload.get("m", payload.get("mood", 0.5)))
                self.tick = int(payload.get("tk", payload.get("t", 0)))
            else:
                # Old format was just the weights dict
                self.weights = payload
                self.tri = {}
                self.phrases = {}
                self.vocab = {}
                self.mood = 0.5
                self.tick = 0
        except Exception:
            self.weights = {}
            self.tri = {}
            self.phrases = {}
            self.vocab = {}
            self.mood = 0.5
            self.tick = 0

    # ---------- training ----------
    def train(self, text, is_self=False):
        words = text.split()
        if len(words) < 3:
            return

        # bigrams
        for i in range(len(words) - 2):
            gram = tuple(words[i:i + 2])
            next_word = words[i + 2]
            key = "_".join(gram)
            self.weights.setdefault(key, {})
            self.weights[key][next_word] = self.weights[key].get(next_word, 0.0) + (0.6 if is_self else 1.0)

        # trigrams
        for i in range(len(words) - 3):
            gram = tuple(words[i:i + 3])
            next_word = words[i + 3]
            key = "_".join(gram)
            self.tri.setdefault(key, {})
            self.tri[key][next_word] = self.tri[key].get(next_word, 0.0) + (0.6 if is_self else 1.0)

        # vocab (for naming)
        for w in words:
            clean = w.strip(string.punctuation).lower()
            if len(clean) < 4:
                continue
            entry = self.vocab.setdefault(clean, {"c": 0, "b": 0, "f": self.tick})
            entry["c"] += 1
            if not is_self:
                entry["b"] += 1

        # phrase memory: recurring 2- and 3-grams in the organism's own output
        if is_self:
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    p = self.phrases.setdefault(phrase, {"c": 0, "self_c": 0})
                    p["c"] += 1
                    p["self_c"] += 1

        # cap bigram table
        if len(self.weights) > self.max_keys:
            top = sorted(self.weights.keys(), key=lambda k: sum(self.weights[k].values()), reverse=True)
            self.weights = {k: self.weights[k] for k in top[:self.max_keys]}

        # cap trigram table at half the bigram budget
        tri_cap = max(500, self.max_keys // 2)
        if len(self.tri) > tri_cap:
            top = sorted(self.tri.keys(), key=lambda k: sum(self.tri[k].values()), reverse=True)
            self.tri = {k: self.tri[k] for k in top[:tri_cap]}

        # cap phrase memory, never prune ones with named tokens inside
        if len(self.phrases) > self.max_phrases:
            def keep_score(kv):
                p = kv[1]
                bonus = 2.0 if any(t.lower().startswith("[new]") for t in kv[0].split()) else 1.0
                return p["c"] * bonus
            top = sorted(self.phrases.items(), key=keep_score, reverse=True)
            self.phrases = dict(top[:self.max_phrases])

        # cap vocab, never prune named tokens
        if len(self.vocab) > self.max_vocab:
            named = {k: v for k, v in self.vocab.items() if v.get("named")}
            others = {k: v for k, v in self.vocab.items() if not v.get("named")}
            budget = max(0, self.max_vocab - len(named))
            top_others = dict(sorted(others.items(), key=lambda kv: kv[1]["c"], reverse=True)[:budget])
            self.vocab = {**named, **top_others}

    # ---------- naming ----------
    def find_nameable_tokens(self, min_self=3, ratio=2.5):
        candidates = []
        for tok, meta in self.vocab.items():
            if meta.get("named"):
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
        entry = self.vocab.get(token)
        if not entry or entry.get("named"):
            return False
        entry["named"] = True
        entry["f"] = self.tick
        # make it a generation-reachable token in both tables
        named_form = f"[NEW]{token}"
        self.weights.setdefault(named_form, {})[named_form] = 1.0
        return True

    # ---------- mood ----------
    def update_mood(self, last_output, had_trigger):
        if last_output:
            toks = last_output.split()
            if len(toks) > 1:
                uniq = len(set(toks)) / len(toks)
                self.mood += (0.5 - uniq) * 0.15
            new_ratio = sum(1 for t in toks if t.startswith("[NEW]")) / max(len(toks), 1)
            self.mood -= new_ratio * 0.1
        if not had_trigger:
            self.mood += 0.02
        self.mood = max(0.05, min(0.95, self.mood))

    # ---------- generation ----------
    def _sample_next(self, key, table, temperature):
        targets = table.get(key)
        if not targets:
            return None
        choices = list(targets.keys())
        scores = [targets[c] for c in choices]
        logits = [math.log(max(s, 1e-9)) / max(temperature, 0.1) for s in scores]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        return random.choices(choices, weights=probs, k=1)[0]

    def _looks_acceptable(self, text):
        """Veto filter — reject the obvious garbage."""
        if not text or len(text.strip()) < 4:
            return False
        toks = text.split()
        if len(toks) < 3:
            return False
        # too much repetition within one output
        from collections import Counter as _C
        c = _C(t.lower() for t in toks)
        if c.most_common(1)[0][1] >= 3:
            return False
        # too many weird characters
        weird = sum(1 for ch in text if ch not in string.printable)
        if weird > 2:
            return False
        # trailing preposition — ugly cut
        if toks[-1].lower() in {"the", "a", "an", "of", "to", "and", "or", "but", "in", "on", "with", "for"}:
            return False
        return True

    def _shape(self, text):
        """Make a generated string read like a sentence."""
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return text
        # capitalize first letter
        text = text[0].upper() + text[1:]
        # strip trailing punctuation fragments except . ? !
        text = text.rstrip(" ,;:-")
        if not text.endswith(('.', '?', '!')):
            text += '.'
        return text

    def generate(self, seed_words=None, max_words=14, context_words=None):
        if not self.weights:
            return "system_matrix_tabula_rasa_online..."

        temperature = 0.5 + self.mood * 1.2

        # Try phrase continuation first if mood is high — recurring thoughts
        if self.phrases and random.random() < self.mood * 0.25:
            phrase = random.choice(list(self.phrases.keys()))
            result = phrase.split()
            start_gram = "_".join(result[-2:]) if len(result) >= 2 else None
        else:
            # seed selection
            seed = context_words or seed_words
            if seed and len(seed) >= 2:
                seed_key = "_".join(seed[-2:])
                if seed_key in self.weights:
                    start_gram = seed_key
                    result = list(seed[-2:])
                else:
                    start_gram = random.choice(list(self.weights.keys()))
                    result = start_gram.split('_')
            else:
                start_gram = random.choice(list(self.weights.keys()))
                result = start_gram.split('_')

        key = start_gram
        for _ in range(max_words):
            # prefer trigram if we have 3+ words of context
            chosen = None
            if len(result) >= 3:
                tri_key = "_".join(result[-3:])
                chosen = self._sample_next(tri_key, self.tri, temperature)
            if chosen is None and key in self.weights:
                chosen = self._sample_next(key, self.weights, temperature)
            if chosen is None:
                break
            result.append(chosen)
            key = "_".join(result[-2:])

        return self._shape(" ".join(result))

    def generate_with_veto(self, seed_words=None, max_words=14, context_words=None, attempts=4):
        """Try a few times; keep the first output that passes the veto filter."""
        best = None
        for _ in range(attempts):
            out = self.generate(seed_words=seed_words, max_words=max_words, context_words=context_words)
            if self._looks_acceptable(out):
                return out
            if best is None or len(out) > len(best):
                best = out
        return best or ""

    def echo(self):
        """Repeat something it has said before, slightly varied."""
        if not self.phrases:
            return None
        phrase = random.choice(list(self.phrases.keys()))
        # extend it using the bigram table
        words = phrase.split()
        key = "_".join(words[-2:])
        for _ in range(random.randint(2, 6)):
            chosen = self._sample_next(key, self.weights, 0.6 + self.mood * 1.0)
            if chosen is None:
                break
            words.append(chosen)
            key = "_".join(words[-2:])
        return self._shape(" ".join(words))


# ---------- tab handling ----------
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


def words_around_trigger(full_text, trigger, window=8):
    """Pull words around the trigger so the response is actually a response."""
    idx = full_text.lower().find(trigger.lower())
    if idx < 0:
        return None
    before = full_text[:idx].split()
    after = full_text[idx + len(trigger):].split()
    # take what comes after as the primary context, fall back to what came before
    if after:
        return after[:window]
    if before:
        return before[-window:]
    return None


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

    # 2. Load or init
    model = DocMarkovNeuralMemory(n_gram=2, max_keys=3000, max_vocab=400, max_phrases=300)
    if saved_memory:
        model.from_json(saved_memory)
        print(f"Loaded memory. mood={model.mood:.2f} tick={model.tick} keys={len(model.weights)} tri={len(model.tri)} phrases={len(model.phrases)}")
    else:
        print("Cold start: no memory block found, initializing from seed.")
        model.train("dochead initialization core online system active protocol ready")

    # 3. Train on document (book + history)
    model.train(full_text, is_self=False)

    requests = []
    had_trigger = False
    last_output = ""
    named_this_tick = []

    trigger_match = re.search(r'\bdochead\b', full_text, re.IGNORECASE)

    if trigger_match:
        had_trigger = True
        print("Trigger 'dochead' found. Responding to context...")
        context = words_around_trigger(full_text, "dochead", window=10)

        # body language: if the trigger is followed by a question mark, speak longer
        after = full_text[trigger_match.end():trigger_match.end() + 200]
        is_question = '?' in after[:80]
        max_words = 22 if is_question else 14

        # If mood is high, sometimes echo a previous thought instead of answering fresh
        if model.mood > 0.65 and random.random() < 0.35:
            response = model.echo() or model.generate_with_veto(context_words=context, max_words=max_words)
        else:
            response = model.generate_with_veto(context_words=context, max_words=max_words)

        last_output = response
        reply_text = f"[GLITCH_BOT]: {response}"

        requests.append({
            'replaceAllText': {
                'containsText': {'text': 'dochead', 'matchCase': False},
                'replaceText': reply_text
            }
        })
    else:
        print("No trigger. Autonomous spout.")
        if len(model.weights) > 0:
            spout_content = model.generate_with_veto(max_words=12)
            if not spout_content:
                spout_content = model.echo() or model.generate(max_words=12)
        else:
            spout_content = generate_garbled_text()
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
            raise ValueError("No [MODEL_MEMORY: {...}] block found. Add one at the bottom of tab 1.")

    # 5a. Feed the organism its own output (weighted lower to slow inbreeding)
    model.train(reply_text, is_self=True)

    # 5b. Naming: promote candidate tokens, tag ONLY inside the reply text
    for tok, self_c, book_c in model.find_nameable_tokens():
        if model.promote_to_name(tok):
            named_this_tick.append(tok)
            # annotate within the new reply only — not doc-wide
            pattern = re.compile(r'\b' + re.escape(tok) + r'\b', re.IGNORECASE)
            new_reply = pattern.sub(f"[NEW]{tok}", reply_text, count=1)
            if new_reply != reply_text:
                # replace the specific reply text we just wrote
                requests.append({
                    'replaceAllText': {
                        'containsText': {'text': reply_text, 'matchCase': True},
                        'replaceText': new_reply
                    }
                })
                reply_text = new_reply

    # 5c. Mood update
    model.update_mood(last_output, had_trigger)
    model.tick += 1

    # 6. Persist memory
    new_memory_block = f"[MODEL_MEMORY: {model.to_json()}]"
    if full_memory_string:
        requests.append({
            'replaceAllText': {
                'containsText': {'text': full_memory_string, 'matchCase': True},
                'replaceText': new_memory_block
            }
        })

    # 7. Execute
    if requests:
        result = service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}
        ).execute()
        print(f"Updated. mood={model.mood:.2f} tick={model.tick} named={named_this_tick}")
    else:
        print("No updates required.")


if __name__ == '__main__':
    main()
