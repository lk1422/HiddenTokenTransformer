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
    "=": 11,
    "<PAD>" : 12,
    "<H>"   : 13,
    "<EOS>" : 14,
    "<SOS>" : 15
}

# Reverse lookup for decoding (optional)
REVERSE_LOOKUP = {v: k for k, v in TOKEN_LOOKUP.items()}
