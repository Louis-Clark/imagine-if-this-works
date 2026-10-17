import os
import random
import json
import re
import math
import string
import hashlib
from collections import Counter, defaultdict
from google.oauth2 import service_account
from googleapiclient.discovery import build

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")

VEC_DIM = 64

# ---------- tiny vector math (works with or without numpy) ----------
def _dot(a, b):
    if HAS_NUMPY:
        return float(np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))

def _norm(a):
    if HAS_NUMPY:
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else a
    n = math.sqrt(sum(x * x for x in a))
    return [x / n if n > 0 else 0.0 for x in a] if n > 0 else a

def _add(a, b, w=1.0):
    if HAS_NUMPY:
        return a + b * w
    return [x + y * w for x, y in zip(a, b)]

def _scale(a, s):
    if HAS_NUMPY:
        return a * s
    return [x * s for x in a]

def _zeros(d=VEC_DIM):
    return np.zeros(d) if HAS_NUMPY else [0.0] * d

def _hash_vector(word, dim=VEC_DIM):
    """Deterministic fallback when numpy is unavailable."""
    h = hashlib.sha256(word.encode()).digest()
    vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)]
    return _norm(vec)

def _to_list(v):
    if HAS_NUMPY:
        return [round(float(x), 4) for x in v]
    return [round(float(x), 4) for x in v]

def _from_list(lst):
    if HAS_NUMPY:
        return np.array(lst, dtype=float)
    return list(lst)

# ---------- PPMI-SVD encoder ----------
class Encoder:
    def __init__(self, dim=VEC_DIM, window=4, min_count=2):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.vocab = {}          # word -> index
        self.vectors = {}        # word -> vector
        self.directions = {}     # name -> vector

    def fit(self, text):
        words = [w.strip(string.punctuation).lower() for w in text.split()]
        words = [w for w in words if len(w) >= 3]
        counts = Counter(words)
        vocab_words = [w for w, c in counts.most_common() if c >= self.min_count]

        if not vocab_words:
            return

        self.vocab = {w: i for i, w in enumerate(vocab_words)}
        V = len(vocab_words)

        # co-occurrence
        co = defaultdict(float)
        for i, w in enumerate(words):
            if w not in self.vocab:
                continue
            wi = self.vocab[w]
            lo = max(0, i - self.window)
            hi = min(len(words), i + self.window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                ctx = words[j]
                if ctx not in self.vocab:
                    continue
                cj = self.vocab[ctx]
                co[(wi, cj)] += 1.0

        if not co:
            return

        # PPMI
        total = sum(co.values())
        row_sums = defaultdict(float)
        col_sums = defaultdict(float)
        for (i, j), c in co.items():
            row_sums[i] += c
            col_sums[j] += c

        if HAS_NUMPY:
            mat = np.zeros((V, V), dtype=float)
            for (i, j), c in co.items():
                pmi = math.log((c * total) / (row_sums[i] * col_sums[j] + 1e-9) + 1e-9)
                mat[i, j] = max(0.0, pmi)
            try:
                U, S, _ = np.linalg.svd(mat, full_matrices=False)
                k = min(self.dim, U.shape[1])
                emb = U[:, :k] * np.sqrt(S[:k])
                if emb.shape[1] < self.dim:
                    pad = np.zeros((emb.shape[0], self.dim - emb.shape[1]))
                    emb = np.hstack([emb, pad])
            except Exception:
                emb = np.random.randn(V, self.dim) * 0.1
            for w, i in self.vocab.items():
                self.vectors[w] = _norm(emb[i])
        else:
            # fallback: hash-based vectors, no semantics but runs
            for w in self.vocab:
                self.vectors[w] = _hash_vector(w, self.dim)

    def embed_word(self, w):
        w = w.strip(string.punctuation).lower()
        if w in self.vectors:
            return self.vectors[w]
        return _hash_vector(w, self.dim)

    def embed(self, text):
        toks = [t.strip(string.punctuation).lower() for t in text.split()]
        toks = [t for t in toks if t]
        if not toks:
            return _zeros()
        vecs = [self.embed_word(t) for t in toks]
        if HAS_NUMPY:
            return _norm(np.mean(vecs, axis=0))
        total = [0.0] * self.dim
        for v in vecs:
            for i in range(self.dim):
                total[i] += v[i]
        return _norm([x / len(vecs) for x in total])

    def build_directions(self):
        seeds = {
            "what":   "what which who define meaning",
            "how":    "how method process steps way",
            "why":    "why because reason cause",
            "yesno":  "yes no is are do does can",
            "invoke": "say speak tell answer",
            "cold":   "cold distant formal detached",
            "warm":   "warm kind close gentle",
            "urgent": "urgent fast now immediate",
            "calm":   "calm slow quiet patient",
        }
        for name, text in seeds.items():
            self.directions[name] = self.embed(text)

    def project(self, vec, direction_name):
        d = self.directions.get(direction_name)
        if d is None:
            return 0.0
        return _dot(_norm(vec), _norm(d))


def cosine(a, b):
    return _dot(_norm(a), _norm(b))


# ---------- serialization helpers for editable blocks ----------
def parse_vectors_block(text):
    """Parse === VECTORS === lines: name | x y z ..."""
    vectors = {}
    m = re.search(r'=== VECTORS ===(.*?)=== END VECTORS ===', text, re.DOTALL)
    if not m:
        return vectors
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or '|' not in line:
            continue
        try:
            name, rest = line.split('|', 1)
            nums = [float(x) for x in rest.split() if x.strip()]
            if nums:
                vectors[name.strip()] = _from_list(nums[:VEC_DIM] if len(nums) >= VEC_DIM else nums + [0.0] * (VEC_DIM - len(nums)))
        except Exception:
            continue
    return vectors


def render_vectors_block(vectors, max_lines=800):
    lines = ["=== VECTORS ==="]
    items = list(vectors.items())[:max_lines]
    for name, vec in items:
        nums = " ".join(f"{x:.3f}" for x in _to_list(vec))
        lines.append(f"{name} | {nums}")
    lines.append("=== END VECTORS ===")
    return "\n".join(lines)


def parse_facts_block(text):
    facts = {}
    m = re.search(r'=== FACTS ===(.*?)=== END FACTS ===', text, re.DOTALL)
    if not m:
        return facts
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or '|' not in line:
            continue
        try:
            body, weight = line.rsplit('|', 1)
            facts[body.strip()] = float(weight.strip())
        except Exception:
            continue
    return facts


def render_facts_block(facts):
    lines = ["=== FACTS ==="]
    for k, v in sorted(facts.items(), key=lambda kv: kv[1], reverse=True)[:200]:
        lines.append(f"{k} | {v:.3f}")
    lines.append("=== END FACTS ===")
    return "\n".join(lines)


def parse_influences_block(text):
    influences = {"ppm": 0.40, "embedding": 0.25, "retrieval": 0.20, "facts": 0.10, "self": 0.05}
    m = re.search(r'=== INFLUENCES ===(.*?)=== END INFLUENCES ===', text, re.DOTALL)
    if not m:
        return influences
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or '|' not in line:
            continue
        try:
            k, v = line.split('|', 1)
            influences[k.strip()] = float(v.strip())
        except Exception:
            continue
    return influences


def render_influences_block(influences):
    lines = ["=== INFLUENCES ==="]
    for k, v in influences.items():
        lines.append(f"{k} | {v:.3f}")
    lines.append("=== END INFLUENCES ===")
    return "\n".join(lines)


def parse_mood_block(text):
    mood = {"valence": 0.0, "arousal": 0.5, "coherence": 0.5, "novelty": 0.5}
    m = re.search(r'=== MOOD ===(.*?)=== END MOOD ===', text, re.DOTALL)
    if not m:
        return mood
    for line in m.group(1).splitlines():
        line = line.strip()
        if '|' in line:
            k, v = line.split('|', 1)
            try:
                mood[k.strip()] = float(v.strip())
            except Exception:
                pass
        elif ':' in line:
            k, v = line.split(':', 1)
            try:
                mood[k.strip()] = float(v.strip())
            except Exception:
                pass
    return mood


def render_mood_block(mood):
    lines = ["=== MOOD ==="]
    for k, v in mood.items():
        lines.append(f"{k} | {v:.3f}")
    lines.append("=== END MOOD ===")
    return "\n".join(lines)


def parse_effects_block(text):
    effects = {}
    m = re.search(r'=== EFFECTS ===(.*?)=== END EFFECTS ===', text, re.DOTALL)
    if not m:
        return effects
    for line in m.group(1).splitlines():
        line = line.strip()
        if '|' in line:
            k, v = line.split('|', 1)
            try:
                effects[k.strip()] = float(v.strip())
            except Exception:
                pass
    return effects


def render_effects_block(effects):
    lines = ["=== EFFECTS ==="]
    for k, v in list(effects.items())[-20:]:
        lines.append(f"{k} | {v:.3f}")
    lines.append("=== END EFFECTS ===")
    return "\n".join(lines)


def parse_exchanges_block(text):
    """Parse === EXCHANGES === lines: tick | vector | 'question' | 'reply'"""
    exchanges = []
    m = re.search(r'=== EXCHANGES ===(.*?)=== END EXCHANGES ===', text, re.DOTALL)
    if not m:
        return exchanges
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split('|')
            if len(parts) < 3:
                continue
            tick = parts[0].strip()
            vec = _from_list([float(x) for x in parts[1].split()][:VEC_DIM])
            exchanges.append({"tick": tick, "vec": vec, "raw": line})
        except Exception:
            continue
    return exchanges


def render_exchanges_block(exchanges, new_entry=None, max_lines=300):
    lines = ["=== EXCHANGES ==="]
    combined = exchanges + ([new_entry] if new_entry else [])
    for e in combined[-max_lines:]:
        lines.append(e["raw"])
    lines.append("=== END EXCHANGES ===")
    return "\n".join(lines)


# ---------- model class (unchanged core, still the decoder) ----------
class DocMarkovNeuralMemory:
    def __init__(self, n_gram=2, max_keys=3000, max_vocab=400, max_phrases=300):
        self.n_gram = n_gram
        self.max_keys = max_keys
        self.max_vocab = max_vocab
        self.max_phrases = max_phrases
        self.weights = {}
        self.tri = {}
        self.phrases = {}
        self.vocab = {}
        self.mood = 0.5
        self.tick = 0

    def to_json(self):
        return json.dumps({
            "w": self.weights, "t3": self.tri, "p": self.phrases,
            "v": self.vocab, "m": round(self.mood, 4), "tk": self.tick,
        }, separators=(',', ':'))

    def from_json(self, json_str):
        try:
            payload = json.loads(json_str)
            if "w" in payload and isinstance(payload.get("w"), dict):
                self.weights = payload.get("w", {})
                self.tri = payload.get("t3", {})
                self.phrases = payload.get("p", {})
                self.vocab = payload.get("v", {})
                self.mood = float(payload.get("m", 0.5))
                self.tick = int(payload.get("tk", 0))
            else:
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

    def train(self, text, is_self=False):
        words = text.split()
        if len(words) < 3:
            return
        for i in range(len(words) - 2):
            gram = tuple(words[i:i + 2])
            nxt = words[i + 2]
            key = "_".join(gram)
            self.weights.setdefault(key, {})
            self.weights[key][nxt] = self.weights[key].get(nxt, 0.0) + (0.6 if is_self else 1.0)
        for i in range(len(words) - 3):
            gram = tuple(words[i:i + 3])
            nxt = words[i + 3]
            key = "_".join(gram)
            self.tri.setdefault(key, {})
            self.tri[key][nxt] = self.tri[key].get(nxt, 0.0) + (0.6 if is_self else 1.0)
        for w in words:
            clean = w.strip(string.punctuation).lower()
            if len(clean) < 4:
                continue
            entry = self.vocab.setdefault(clean, {"c": 0, "b": 0, "f": self.tick})
            entry["c"] += 1
            if not is_self:
                entry["b"] += 1
        if is_self:
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    p = self.phrases.setdefault(phrase, {"c": 0})
                    p["c"] += 1
        if len(self.weights) > self.max_keys:
            top = sorted(self.weights.keys(), key=lambda k: sum(self.weights[k].values()), reverse=True)
            self.weights = {k: self.weights[k] for k in top[:self.max_keys]}
        tri_cap = max(500, self.max_keys // 2)
        if len(self.tri) > tri_cap:
            top = sorted(self.tri.keys(), key=lambda k: sum(self.tri[k].values()), reverse=True)
            self.tri = {k: self.tri[k] for k in top[:tri_cap]}
        if len(self.phrases) > self.max_phrases:
            top = sorted(self.phrases.items(), key=lambda kv: kv[1]["c"], reverse=True)
            self.phrases = dict(top[:self.max_phrases])
        if len(self.vocab) > self.max_vocab:
            named = {k: v for k, v in self.vocab.items() if v.get("named")}
            others = {k: v for k, v in self.vocab.items() if not v.get("named")}
            budget = max(0, self.max_vocab - len(named))
            top_others = dict(sorted(others.items(), key=lambda kv: kv[1]["c"], reverse=True)[:budget])
            self.vocab = {**named, **top_others}

    def find_nameable_tokens(self, min_self=3, ratio=2.5):
        candidates = []
        for tok, meta in self.vocab.items():
            if meta.get("named") or len(tok) < 5:
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
        self.weights.setdefault(f"[NEW]{token}", {})[f"[NEW]{token}"] = 1.0
        return True

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
        if not text or len(text.strip()) < 4:
            return False
        toks = text.split()
        if len(toks) < 3:
            return False
        c = Counter(t.lower() for t in toks)
        if c.most_common(1)[0][1] >= 3:
            return False
        if sum(1 for ch in text if ch not in string.printable) > 2:
            return False
        if toks[-1].lower() in {"the", "a", "an", "of", "to", "and", "or", "but", "in", "on", "with", "for"}:
            return False
        return True

    def _shape(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return text
        text = text[0].upper() + text[1:]
        text = text.rstrip(" ,;:-")
        if not text.endswith(('.', '?', '!')):
            text += '.'
        return text

    def generate(self, seed_words=None, max_words=14, context_words=None):
        if not self.weights:
            return "system_matrix_tabula_rasa_online..."
        temperature = 0.5 + self.mood * 1.2
        if self.phrases and random.random() < self.mood * 0.25:
            phrase = random.choice(list(self.phrases.keys()))
            result = phrase.split()
            start_gram = "_".join(result[-2:]) if len(result) >= 2 else None
        else:
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
            chosen = None
            if len(result) >= 3:
                chosen = self._sample_next("_".join(result[-3:]), self.tri, temperature)
            if chosen is None and key in self.weights:
                chosen = self._sample_next(key, self.weights, temperature)
            if chosen is None:
                break
            result.append(chosen)
            key = "_".join(result[-2:])
        return self._shape(" ".join(result))

    def generate_with_veto(self, seed_words=None, max_words=14, context_words=None, attempts=4):
        best = None
        for _ in range(attempts):
            out = self.generate(seed_words=seed_words, max_words=max_words, context_words=context_words)
            if self._looks_acceptable(out):
                return out
            if best is None or len(out) > len(best):
                best = out
        return best or ""

    def echo(self):
        if not self.phrases:
            return None
        phrase = random.choice(list(self.phrases.keys()))
        words = phrase.split()
        key = "_".join(words[-2:])
        for _ in range(random.randint(2, 6)):
            chosen = self._sample_next(key, self.weights, 0.6 + self.mood * 1.0)
            if chosen is None:
                break
            words.append(chosen)
            key = "_".join(words[-2:])
        return self._shape(" ".join(words))


# ---------- doc reading ----------
def add_current_and_child_tabs(tab, all_tabs):
    all_tabs.append(tab)
    for child in tab.get('childTabs', []):
        add_current_and_child_tabs(child, all_tabs)


def get_all_tabs(doc):
    all_tabs = []
    for tab in doc.get('tabs', []):
        add_current_and_child_tabs(tab, all_tabs)
    return all_tabs


def extract_all_docs_data(doc):
    full_text = ""
    all_tabs = get_all_tabs(doc)
    for tab in all_tabs:
        body = tab.get('documentTab', {}).get('body', {}).get('content', [])
        for element in body:
            if 'paragraph' in element:
                for elem in element['paragraph'].get('elements', []):
                    if 'textRun' in elem:
                        full_text += elem['textRun'].get('content', '') + "\n"
    match = re.search(r'\[MODEL_MEMORY:\s*(\{.*?\})\s*\]', full_text)
    saved_memory = match.group(1) if match else None
    full_memory_string = match.group(0) if match else None
    return full_text, saved_memory, full_memory_string, all_tabs


def generate_garbled_text():
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;':\",./<>?~`~# "
    return "".join(random.choice(alphabet) for _ in range(random.randint(20, 50)))


def words_around_trigger(full_text, trigger, window=10):
    idx = full_text.lower().find(trigger.lower())
    if idx < 0:
        return None
    after = full_text[idx + len(trigger):].split()
    if after:
        return after[:window]
    before = full_text[:idx].split()
    return before[-window:] if before else None


def extract_question(text, trigger_match):
    """Take text after 'dochead' up to next sentence boundary."""
    if not trigger_match:
        return ""
    tail = text[trigger_match.end():trigger_match.end() + 400]
    tail = re.sub(r'^\s*[:\-\u2014]\s*', '', tail)
    m = re.match(r'([^.?!]{3,200}[.?!]?)', tail.strip())
    return m.group(1).strip() if m else ""


def classify_question(encoder, q_text):
    qv = encoder.embed(q_text)
    scores = {name: encoder.project(qv, name) for name in ("what", "how", "why", "yesno", "invoke")}
    q_type = max(scores, key=scores.get)
    register_scores = {name: encoder.project(qv, name) for name in ("cold", "warm", "urgent", "calm")}
    register = max(register_scores, key=register_scores.get)
    return q_type, register, qv, scores


def nearest_subject(encoder, q_vec, vectors, k=1):
    best = []
    for name, vec in vectors.items():
        if not name.startswith("word:") and not name.startswith("fact:"):
            continue
        sim = cosine(q_vec, vec)
        best.append((sim, name))
    best.sort(reverse=True)
    return best[:k]


def blend_vectors(items):
    """items: list of (vec, weight). Returns normalized blend."""
    if not items:
        return _zeros()
    total_w = sum(w for _, w in items if w > 0)
    if total_w <= 0:
        return _zeros()
    acc = _zeros()
    for vec, w in items:
        if w <= 0:
            continue
        acc = _add(acc, vec, w / total_w)
    return _norm(acc)


def decode_blend_to_seed_words(encoder, blend, vectors, max_words=3):
    """Turn a vector blend into a small set of seed words for the markov decoder."""
    best = []
    for name, vec in vectors.items():
        if name.startswith("word:"):
            sim = cosine(blend, vec)
            best.append((sim, name[5:]))
    best.sort(reverse=True)
    return [w for _, w in best[:max_words]]


def extract_facts_from_reply(reply_text):
    """Very simple: subject is X, subject has X -> collect predicate phrases."""
    facts = []
    patterns = [
        r'(\b\w[\w\s]{2,20})\s+is\s+([\w\s]{2,40})',
        r'(\b\w[\w\s]{2,20})\s+has\s+([\w\s]{2,40})',
        r'(\b\w[\w\s]{2,20})\s+was\s+([\w\s]{2,40})',
    ]
    for pat in patterns:
        for m in re.finditer(pat, reply_text, re.IGNORECASE):
            subj = m.group(1).strip()
            pred = m.group(2).strip().rstrip('.,;:')
            if 2 < len(subj) < 30 and 2 < len(pred) < 50:
                facts.append(f"{subj} is {pred}" if ' is ' in m.group(0).lower() else
                             f"{subj} has {pred}" if ' has ' in m.group(0).lower() else
                             f"{subj} was {pred}")
    return facts[:3]


def main():
    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    document_id = os.environ.get("GOOGLE_DOC_ID")
    if not service_account_json or not document_id:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID env vars.")

    creds_info = service_account.Credentials.from_service_account_info(
        eval(service_account_json),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    doc = service.documents().get(documentId=document_id, includeTabsContent=True).execute()
    full_text, saved_memory, full_memory_string, all_tabs = extract_all_docs_data(doc)

    # ---- load structured blocks ----
    vectors = parse_vectors_block(full_text)
    facts = parse_facts_block(full_text)
    influences = parse_influences_block(full_text)
    mood = parse_mood_block(full_text)
    exchanges = parse_exchanges_block(full_text)
    effects = parse_effects_block(full_text)

    # ---- load or init markov decoder ----
    model = DocMarkovNeuralMemory(n_gram=2, max_keys=3000, max_vocab=400, max_phrases=300)
    if saved_memory:
        model.from_json(saved_memory)
        print(f"Loaded memory. tick={model.tick} mood={model.mood:.2f}")
    else:
        model.train("dochead initialization core online system active protocol ready")

    # ---- fit encoder on doc ----
    encoder = Encoder(dim=VEC_DIM, window=4, min_count=2)
    encoder.fit(full_text)
    encoder.build_directions()

    # Merge newly-learned word vectors into the stored vectors table (doc wins on conflict)
    for w, vec in encoder.vectors.items():
        key = f"word:{w}"
        if key not in vectors:
            vectors[key] = vec

    # Add fact vectors
    for fact_text in list(facts.keys()):
        key = f"fact:{fact_text}"
        if key not in vectors:
            vectors[key] = encoder.embed(fact_text)

    # ---- train decoder ----
    model.train(full_text, is_self=False)

    requests = []
    had_trigger = False
    last_output = ""
    named_this_tick = []
    new_exchange_raw = None

    trigger_match = re.search(r'\bdochead\b', full_text, re.IGNORECASE)

    # ---- conversational response ----
    if trigger_match:
        had_trigger = True
        question = extract_question(full_text, trigger_match)
        print(f"Trigger found. Question: '{question[:80]}'")

        if question:
            q_type, q_register, q_vec, q_scores = classify_question(encoder, question)
            subject_hits = nearest_subject(encoder, q_vec, vectors, k=1)
            subject_name = subject_hits[0][1] if subject_hits else None
            subject_vec = vectors.get(subject_name) if subject_name else _zeros()

            # novelty: how close is this question to a past exchange?
            novelty = 0.7
            retrieved = []
            if exchanges:
                scored = [(cosine(q_vec, e["vec"]), e) for e in exchanges[-200:]]
                scored.sort(reverse=True, key=lambda x: x[0])
                retrieved = [e for sim, e in scored[:2] if sim > 0.3]
                novelty = 1.0 - max((sim for sim, _ in scored[:1]), default=0.0)

            # relevant facts
            relevant_facts = [(f, w) for f, w in facts.items() if subject_name and subject_name[5:] in f.lower()][:3]

            # build blend
            blend_items = [
                (q_vec, 1.0),
                (subject_vec, influences["embedding"] * 2.0),
            ]
            for e in retrieved:
                blend_items.append((e["vec"], influences["retrieval"]))
            for fact_text, weight in relevant_facts:
                fact_vec = vectors.get(f"fact:{fact_text}")
                if fact_vec is not None:
                    blend_items.append((fact_vec, influences["facts"] * weight))

            blend = blend_vectors(blend_items)

            # apply mood vector offset
            mood_vec = blend_vectors([
                (vectors.get("dir:warm", _zeros()),   mood.get("valence", 0)),
                (vectors.get("dir:urgent", _zeros()), mood.get("arousal", 0)),
                (q_vec, mood.get("novelty", 0.5) * 0.1),
            ])
            blend = _norm(_add(blend, mood_vec, 0.25))

            # seed words for the markov decoder
            seed_words = decode_blend_to_seed_words(encoder, blend, vectors, max_words=3)

            max_words = 22 if q_type in ("why",) or '?' in question else 14
            if q_type == "yesno":
                max_words = 10

            response = model.generate_with_veto(context_words=seed_words, max_words=max_words)
            if not response:
                response = model.generate(max_words=max_words)

            print(f"  type={q_type} register={q_register} novelty={novelty:.2f} seed={seed_words}")
        else:
            # trigger with no question text — pure continuation
            response = model.generate_with_veto(
                context_words=words_around_trigger(full_text, "dochead", window=10),
                max_words=14
            )
            q_vec = encoder.embed("")

        last_output = response
        reply_text = f"[GLITCH_BOT]: {response}"

        requests.append({
            'replaceAllText': {
                'containsText': {'text': 'dochead', 'matchCase': False},
                'replaceText': reply_text
            }
        })

        # record the exchange vector for future retrieval
        try:
            r_vec = encoder.embed(response)
            exch_vec = _norm(_add(q_vec, r_vec, 1.0))
            new_exchange_raw = (
                f"{model.tick} | " + " ".join(f"{x:.3f}" for x in _to_list(exch_vec)) +
                f" | {question[:60]!r} | {response[:60]!r}"
            )
        except Exception:
            new_exchange_raw = None

        # extract facts from reply
        for f in extract_facts_from_reply(response):
            facts[f] = min(1.0, facts.get(f, 0.0) + 0.3)
            fk = f"fact:{f}"
            if fk not in vectors:
                vectors[fk] = encoder.embed(f)

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

    # ---- self-training ----
    model.train(reply_text, is_self=True)

    # ---- naming (only in this reply) ----
    for tok, self_c, book_c in model.find_nameable_tokens():
        if model.promote_to_name(tok):
            named_this_tick.append(tok)
            pattern = re.compile(r'\b' + re.escape(tok) + r'\b', re.IGNORECASE)
            new_reply = pattern.sub(f"[NEW]{tok}", reply_text, count=1)
            if new_reply != reply_text:
                requests.append({
                    'replaceAllText': {
                        'containsText': {'text': reply_text, 'matchCase': True},
                        'replaceText': new_reply
                    }
                })
                reply_text = new_reply

    # ---- mood update (scalar for markov + vector for doc) ----
    model.update_mood(last_output, had_trigger)
    model.tick += 1

    # update mood vector
    if last_output:
        toks = last_output.split()
        if len(toks) > 1:
            uniq = len(set(toks)) / len(toks)
            mood["novelty"] = min(0.95, max(0.05, mood.get("novelty", 0.5) + (0.6 - uniq) * 0.1))
        mood["coherence"] = min(0.95, max(0.05, 0.5 + (model.mood - 0.5) * 0.8))
    mood["arousal"] = min(0.95, max(0.05, mood.get("arousal", 0.5) + (0.02 if not had_trigger else -0.01)))

    # ---- effects: self-modification of influences ----
    effects_delta = {}
    if had_trigger and new_exchange_raw:
        effects_delta["retrieval_weight"] = influences["retrieval"] * 0.05
        influences["retrieval"] = min(0.6, influences["retrieval"] + effects_delta["retrieval_weight"])
    if named_this_tick:
        effects_delta["novelty"] = 0.02
        mood["novelty"] = min(0.95, mood["novelty"] + 0.02)
    for k, v in effects_delta.items():
        effects[f"tick_{model.tick}_{k}"] = v

    # ---- decay facts slightly ----
    for f in list(facts.keys()):
        facts[f] *= 0.995
        if facts[f] < 0.05:
            del facts[f]

    # ---- persist all blocks ----
    new_vectors_block = render_vectors_block(vectors, max_lines=600)
    new_facts_block = render_facts_block(facts)
    new_influences_block = render_influences_block(influences)
    new_mood_block = render_mood_block(mood)
    new_effects_block = render_effects_block(effects)
    new_exchanges_block = render_exchanges_block(exchanges, new_entry={"raw": new_exchange_raw} if new_exchange_raw else None)

    # replace each block in the doc; if a block doesn't exist, append it
    for old_header, new_block in [
        ("=== VECTORS ===", new_vectors_block),
        ("=== FACTS ===", new_facts_block),
        ("=== INFLUENCES ===", new_influences_block),
        ("=== MOOD ===", new_mood_block),
        ("=== EFFECTS ===", new_effects_block),
        ("=== EXCHANGES ===", new_exchanges_block),
    ]:
        pattern = re.compile(re.escape(old_header) + r'.*?=== END ' + old_header.strip('= ').strip() + r' ===', re.DOTALL)
        m = pattern.search(full_text)
        if m:
            requests.append({
                'replaceAllText': {
                    'containsText': {'text': m.group(0), 'matchCase': True},
                    'replaceText': new_block
                }
            })
        else:
            # Append the block near the memory tag
            requests.append({
                'replaceAllText': {
                    'containsText': {'text': full_memory_string or '[MODEL_MEMORY:', 'matchCase': False},
                    'replaceText': new_block + "\n" + (full_memory_string or '')
                }
            })

    # ---- persist markov memory ----
    new_memory_block = f"[MODEL_MEMORY: {model.to_json()}]"
    if full_memory_string:
        requests.append({
            'replaceAllText': {
                'containsText': {'text': full_memory_string, 'matchCase': True},
                'replaceText': new_memory_block
            }
        })

    if requests:
        result = service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests}
        ).execute()
        print(f"Updated. tick={model.tick} mood={model.mood:.2f} named={named_this_tick}")
    else:
        print("No updates.")


if __name__ == '__main__':
    main()
