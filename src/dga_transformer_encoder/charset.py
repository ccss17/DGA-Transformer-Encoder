import string

# Character-level tokenizer for registrable domain labels (no TLD).
# Allowed set covers LDH + underscore/specials rarely seen; unknown chars map to PAD.

CHARS = list(string.ascii_lowercase + string.digits + "-_")

SPECIAL_TOKENS = ("<pad>", "<cls>")
PAD, CLS = range(len(SPECIAL_TOKENS))
SPECIAL_OFFSET = len(SPECIAL_TOKENS)

stoi = {c: i + SPECIAL_OFFSET for i, c in enumerate(CHARS)}  # reserve space for PAD/CLS
itos = {i: c for c, i in stoi.items()}
VOCAB_SIZE = SPECIAL_OFFSET + len(CHARS)  # include special tokens
ALLOWED_CHARS = set(stoi)

def normalize_domain(d: str) -> str:
    """ex) normalize_domain("Example_Domain!") -> "example_domain"
    """
    d = (d or "").strip().lower()
    return "".join(ch for ch in d if ch in ALLOWED_CHARS)

def encode_domain(d: str, max_len: int = 64):
    """Convert a domain label into fixed-length token ids.

    ex) encode_domain("abc", max_len=5) -> [1, 2, 3, 4, 0]
    ex) encode_domain("abcdef", max_len=4) -> [1, 2, 3, 4]
    """
    d = normalize_domain(d)
    ids = [CLS] + [stoi.get(ch, PAD) for ch in d][: max_len - 1]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return ids
