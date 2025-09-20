# Character-level tokenizer for registrable domain labels (no TLD).
# Allowed set covers LDH + underscore/specials rarely seen; unknown chars map to PAD.

CHARS = (
    list("abcdefghijklmnopqrstuvwxyz") +
    list("0123456789") +
    list("-_")  # LDH core (underscore rarely appears in domains but kept for safety)
)

PAD, CLS = 0, 1
stoi = {c: i + 2 for i, c in enumerate(CHARS)}  # 0:PAD, 1:CLS
itos = {i: c for c, i in stoi.items()}

VOCAB_SIZE = len(stoi) + 2  # include PAD + CLS

def normalize_domain(d: str) -> str:
    """Lowercase, strip spaces, keep punycode prefix 'xn--' as is. TLD should be removed upstream.
    Non-allowed chars are dropped here to keep the tokenizer simple."""
    d = (d or "").strip().lower()
    # keep only allowed chars
    filtered = []
    for ch in d:
        if ch in stoi:
            filtered.append(ch)
        # silently drop others
    return "".join(filtered)

def encode_domain(d: str, max_len: int = 64):
    d = normalize_domain(d)
    ids = [CLS] + [stoi.get(ch, PAD) for ch in d][: max_len - 1]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return ids