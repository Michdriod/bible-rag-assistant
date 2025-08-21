import re
# Define aliases for Bible book names
BIBLE_BOOK_ALIASES = {
    "Genesis": ["Gen", "Gn"],
    "Exodus": ["Exod", "Ex"],
    "Leviticus": ["Lev", "Lv"],
    "Numbers": ["Num", "Nm"],
    "Deuteronomy": ["Deut", "Dt"],
    "Joshua": ["Josh", "Jos"],
    "Judges": ["Judg", "Jdg"],
    "Ruth": ["Rth", "Ru"],
    "1 Samuel": ["1 Sam", "I Sam", "1Sm", "1Sa"],
    "2 Samuel": ["2 Sam", "II Sam", "2Sm", "2Sa"],
    "1 Kings": ["1 Kgs", "I Kgs", "1Ki", "1Kg"],
    "2 Kings": ["2 Kgs", "II Kgs", "2Ki", "2Kg"],
    "1 Chronicles": ["1 Chr", "I Chr", "1Ch", "1Chron"],
    "2 Chronicles": ["2 Chr", "II Chr", "2Ch", "2Chron"],
    "Ezra": ["Ezr"],
    "Nehemiah": ["Neh", "Ne"],
    "Esther": ["Est", "Es"],
    "Job": ["Job"],
    "Psalms": ["Ps", "Pslm", "Psa", "Psalm"],
    "Proverbs": ["Prov", "Prv"],
    "Ecclesiastes": ["Eccl", "Ecc"],
    "Song of Solomon": ["Song", "Song of Songs", "SOS", "Canticles"],
    "Isaiah": ["Isa", "Is"],
    "Jeremiah": ["Jer", "Jr"],
    "Lamentations": ["Lam", "La"],
    "Ezekiel": ["Ezek", "Eze"],
    "Daniel": ["Dan", "Dn"],
    "Hosea": ["Hos", "Ho"],
    "Joel": ["Joel", "Jl"],
    "Amos": ["Am", "Amo"],
    "Obadiah": ["Obad", "Ob"],
    "Jonah": ["Jon", "Jnh"],
    "Micah": ["Mic", "Mc"],
    "Nahum": ["Nah", "Na"],
    "Habakkuk": ["Hab", "Hb"],
    "Zephaniah": ["Zeph", "Zep", "Zp"],
    "Haggai": ["Hag", "Hg"],
    "Zechariah": ["Zech", "Zec", "Zc"],
    "Malachi": ["Mal", "Ml"],
    "Matthew": ["Matt", "Mt"],
    "Mark": ["Mk", "Mrk"],
    "Luke": ["Lk", "Lu"],
    "John": ["Jn", "Jhn"],
    "Acts": ["Ac", "Acts"],
    "Romans": ["Rom", "Rm"],
    "1 Corinthians": ["1 Cor", "I Cor", "1Co"],
    "2 Corinthians": ["2 Cor", "II Cor", "2Co"],
    "Galatians": ["Gal", "Ga"],
    "Ephesians": ["Eph", "Ep"],
    "Philippians": ["Phil", "Php", "Phl"],
    "Colossians": ["Col", "Co"],
    "1 Thessalonians": ["1 Thess", "I Thess", "1Th"],
    "2 Thessalonians": ["2 Thess", "II Thess", "2Th"],
    "1 Timothy": ["1 Tim", "I Tim", "1Ti"],
    "2 Timothy": ["2 Tim", "II Tim", "2Ti"],
    "Titus": ["Tit", "Ti"],
    "Philemon": ["Philem", "Phm"],
    "Hebrews": ["Heb"],
    "James": ["Jas", "Jm"],
    "1 Peter": ["1 Pet", "I Pet", "1Pe"],
    "2 Peter": ["2 Pet", "II Pet", "2Pe"],
    "1 John": ["1 Jn", "I Jn", "1Jo"],
    "2 John": ["2 Jn", "II Jn", "2Jo"],
    "3 John": ["3 Jn", "III Jn", "3Jo"],
    "Jude": ["Jud", "Jude"],
    "Revelation": ["Rev", "Rv"]
}


# Create lowercase set of all book names and their aliases
ALL_BOOK_NAMES = set()
for book, aliases in BIBLE_BOOK_ALIASES.items():
    ALL_BOOK_NAMES.add(book.lower())
    for alias in aliases:
        ALL_BOOK_NAMES.add(alias.lower())


def is_valid_bible_query(query: str) -> bool:
    """
    Determine if a given query is likely a Bible-related query.
    
    Checks for:
    - Valid Bible references (e.g., John 3:16, 1 Cor 13:4)
    - Book names or aliases
    - Presence of spiritual terms
    """
    query = query.strip().lower()

    # Match references like "John 3:16" or "1 Cor 13:4"
    reference_pattern = re.compile(r'\b(?:[1-3]?\s?[A-Za-z]{2,})\s+\d{1,3}:\d{1,3}\b')
    if reference_pattern.search(query):
        return True

    # Check for book name or alias presence
    for name in ALL_BOOK_NAMES:
        if name in query:
            return True

    # Check for spiritual/religious keywords
    if re.search(r'\b(God|Jesus|Holy|Spirit|Christ|Heaven|Lord|Faith|Pray|Sin|Cross|Grace|Mercy)\b', query, re.IGNORECASE):
        return True

    return False









# # A small sample of Bible book names to use in matching
# BIBLE_BOOKS = [
#     "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
#     "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "Corinthians",
#     "Revelation", "Psalms", "Proverbs", "Isaiah", "Jeremiah"
#     # ... you can expand this
# ]

# def is_valid_bible_query(query: str) -> bool:
#     query = query.strip()

#     # Check for pattern like "John 3:16" or "1 Corinthians 13:4"
#     bible_reference_pattern = re.compile(r'\b(?:[1-3]?\s?[A-Za-z]+)\s+\d{1,3}:\d{1,3}\b')
#     if bible_reference_pattern.search(query):
#         return True

#     # Check if the query contains any known Bible book name
#     for book in BIBLE_BOOKS:
#         if book.lower() in query.lower():
#             return True

#     # Check if query contains "God", "Jesus", or other spiritual terms
#     if re.search(r'\b(God|Jesus|Holy|Spirit|Christ|Heaven|Lord|Faith|Pray|Sin|Cross)\b', query, re.IGNORECASE):
#         return True

#     return False

