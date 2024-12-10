# Token lookup dictionary
TOKEN_LOOKUP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "*": 11, 
    "=": 12,
    "<H>": 13,
    "<EOS>": 14,
    "<S>": 15,
    "<PAD>": 16,
}

# Reverse lookup for decoding (optional)
REVERSE_LOOKUP = {v: k for k, v in TOKEN_LOOKUP.items()}
