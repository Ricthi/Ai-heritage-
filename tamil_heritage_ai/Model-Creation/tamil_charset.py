# tamil_charset.py

# Canonical ordered list of all 247 Tamil characters.
# Order: Vowels (12) -> Consonants (18) -> Uyirmei (216) -> Ayutham (1)

UYIR = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"]
MEI = ["க்", "ங்", "ச்", "ஞ்", "ட்", "ண்", "த்", "ந்", "ப்", "ம்", "ய்", "ர்", "ல்", "வ்", "ழ்", "ள்", "ற்", "ன்"]
MEI_BASES = ["க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன"]
VOWEL_MARKERS = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]

UYIRMEI = [base + marker for base in MEI_BASES for marker in VOWEL_MARKERS]
AYUTHAM = ["ஃ"]

TAMIL_CHARS = UYIR + MEI + UYIRMEI + AYUTHAM
NUM_CLASSES = len(TAMIL_CHARS)

# Mapping for fast lookup
CHAR_TO_IDX = {ch: i for i, ch in enumerate(TAMIL_CHARS)}
IDX_TO_CHAR = {i: ch for i, ch in enumerate(TAMIL_CHARS)}

# Legacy mapping (for backwards compatibility if needed in this file)
TAMIL_MAP = {
    'ai': 'ஐ', 'c': 'ச்', 'e': 'எ', 'i': 'இ', 'k': 'க்', 'l': 'ல்', 'l5': 'ள்',
    'l5u': 'லு', 'l5u4': 'லூ', 'n': 'ந்', 'n1': 'ன்', 'n1u4': 'னூ', 'n2': 'ண்',
    'n2u4': 'ணூ', 'n3': 'ங்', 'n5': 'ம்', 'o': 'ஒ', 'p': 'ப்', 'pu4': 'பூ',
    'r': 'ர்', 'r5': 'ற்', 'r5i': 'றி', 'ru': 'ரு', 't': 'த்', 'ti': 'தி', 'y': 'ய்'
}
