import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Define data classes for Bible verse chunks and ranges
@dataclass
class BibleVerseChunk:
    book: str
    chapter: int
    verse: int
    text: str
    reference: str

@dataclass
class VerseRange:
    book: str
    chapter: int
    start_verse: int
    end_verse: int
    
    def to_references(self) -> List[str]:
        """Convert range to list of individual references"""
        references = []
        for verse_num in range(self.start_verse, self.end_verse + 1):
            references.append(f"{self.book} {self.chapter}:{verse_num}")
        return references

# Define the BibleChunker class for text chunking and normalization
class BibleChunker:
    """
    Enhanced chunker for Bible text that supports both single verses and ranges
    """
    
    def __init__(self):
        # Bible book mappings for normalization
        self.book_abbreviations = {
            # Old Testament
            "gen": "Genesis", "genesis": "Genesis",
            "exod": "Exodus", "exodus": "Exodus", "ex": "Exodus",
            "lev": "Leviticus", "leviticus": "Leviticus",
            "num": "Numbers", "numbers": "Numbers",
            "deut": "Deuteronomy", "deuteronomy": "Deuteronomy", "dt": "Deuteronomy",
            "josh": "Joshua", "joshua": "Joshua",
            "judg": "Judges", "judges": "Judges",
            "ruth": "Ruth",
            "1sam": "1 Samuel", "1 samuel": "1 Samuel", "1 sam": "1 Samuel",
            "2sam": "2 Samuel", "2 samuel": "2 Samuel", "2 sam": "2 Samuel",
            "1kgs": "1 Kings", "1 kings": "1 Kings", "1 ki": "1 Kings",
            "2kgs": "2 Kings", "2 kings": "2 Kings", "2 ki": "2 Kings",
            "1chr": "1 Chronicles", "1 chronicles": "1 Chronicles", "1 ch": "1 Chronicles",
            "2chr": "2 Chronicles", "2 chronicles": "2 Chronicles", "2 ch": "2 Chronicles",
            "ezra": "Ezra",
            "neh": "Nehemiah", "nehemiah": "Nehemiah",
            "esth": "Esther", "esther": "Esther",
            "job": "Job",
            "ps": "Psalm", "psalm": "Psalm", "psalms": "Psalm", "psa": "Psalm",
            "prov": "Proverbs", "proverbs": "Proverbs", "pr": "Proverbs",
            "eccl": "Ecclesiastes", "ecclesiastes": "Ecclesiastes", "ecc": "Ecclesiastes",
            "song": "Song of Solomon", "songs": "Song of Solomon", "sos": "Song of Solomon",
            "isa": "Isaiah", "isaiah": "Isaiah", "is": "Isaiah",
            "jer": "Jeremiah", "jeremiah": "Jeremiah",
            "lam": "Lamentations", "lamentations": "Lamentations",
            "ezek": "Ezekiel", "ezekiel": "Ezekiel", "eze": "Ezekiel",
            "dan": "Daniel", "daniel": "Daniel",
            "hos": "Hosea", "hosea": "Hosea",
            "joel": "Joel",
            "amos": "Amos",
            "obad": "Obadiah", "obadiah": "Obadiah",
            "jonah": "Jonah",
            "mic": "Micah", "micah": "Micah",
            "nah": "Nahum", "nahum": "Nahum",
            "hab": "Habakkuk", "habakkuk": "Habakkuk",
            "zeph": "Zephaniah", "zephaniah": "Zephaniah", "zep": "Zephaniah",
            "hag": "Haggai", "haggai": "Haggai",
            "zech": "Zechariah", "zechariah": "Zechariah", "zec": "Zechariah",
            "mal": "Malachi", "malachi": "Malachi",
            
            # New Testament
            "matt": "Matthew", "matthew": "Matthew", "mt": "Matthew",
            "mark": "Mark", "mk": "Mark",
            "luke": "Luke", "lk": "Luke",
            "john": "John", "jn": "John", "joh": "John",
            "acts": "Acts", "ac": "Acts",
            "rom": "Romans", "romans": "Romans", "ro": "Romans",
            "1cor": "1 Corinthians", "1 corinthians": "1 Corinthians", "1 cor": "1 Corinthians",
            "2cor": "2 Corinthians", "2 corinthians": "2 Corinthians", "2 cor": "2 Corinthians",
            "gal": "Galatians", "galatians": "Galatians",
            "eph": "Ephesians", "ephesians": "Ephesians",
            "phil": "Philippians", "philippians": "Philippians", "php": "Philippians",
            "col": "Colossians", "colossians": "Colossians",
            "1thess": "1 Thessalonians", "1 thessalonians": "1 Thessalonians", "1 th": "1 Thessalonians",
            "2thess": "2 Thessalonians", "2 thessalonians": "2 Thessalonians", "2 th": "2 Thessalonians",
            "1tim": "1 Timothy", "1 timothy": "1 Timothy", "1 ti": "1 Timothy",
            "2tim": "2 Timothy", "2 timothy": "2 Timothy", "2 ti": "2 Timothy",
            "titus": "Titus", "tit": "Titus",
            "phlm": "Philemon", "philemon": "Philemon", "phm": "Philemon",
            "heb": "Hebrews", "hebrews": "Hebrews",
            "jas": "James", "james": "James", "ja": "James",
            "1pet": "1 Peter", "1 peter": "1 Peter", "1 pe": "1 Peter",
            "2pet": "2 Peter", "2 peter": "2 Peter", "2 pe": "2 Peter",
            "1john": "1 John", "1 jn": "1 John", "1 jo": "1 John",
            "2john": "2 John", "2 jn": "2 John", "2 jo": "2 John",
            "3john": "3 John", "3 jn": "3 John", "3 jo": "3 John",
            "jude": "Jude",
            "rev": "Revelation", "revelation": "Revelation", "re": "Revelation"
        }
    
    def normalize_book_name(self, book: str) -> str:
        """Normalize book name to standard format"""
        book_lower = book.lower().strip()
        return self.book_abbreviations.get(book_lower, book.title())
    
    def is_chapter_reference(self, text: str) -> bool:
        """
        Check if the input text is a chapter-only reference like 'Genesis 12'
        """
        chapter_patterns = [
            r'^[12]?\s*[A-Za-z]+\.?\s+\d+$',               # Genesis 12, 1 Kings 5
            r'^[A-Za-z\s]+\s+\d+$'                         # Song of Solomon 2
        ]
        
        for pattern in chapter_patterns:
            if re.match(pattern, text.strip()):
                return True
        return False
    
    def parse_chapter_reference(self, reference: str) -> Tuple[str, int]:
        """
        Parse a chapter-only Bible reference like 'Genesis 12' into components
        Returns: (book, chapter)
        """
        # Pattern to match: Book Chapter
        pattern = r'^(.+?)\s+(\d+)$'
        match = re.match(pattern, reference.strip())
        
        if not match:
            raise ValueError(f"Invalid chapter reference format: {reference}")
        
        book_name = self.normalize_book_name(match.group(1))
        chapter = int(match.group(2))
        
        return book_name, chapter
    
    def parse_reference(self, reference: str) -> Tuple[str, int, int]:
        """
        Parse a single Bible reference like 'John 3:16' into components
        Returns: (book, chapter, verse)
        """
        # Pattern to match: Book Chapter:Verse
        pattern = r'^(.+?)\s+(\d+):(\d+)$'
        match = re.match(pattern, reference.strip())
        
        if not match:
            raise ValueError(f"Invalid Bible reference format: {reference}")
        
        book_name = self.normalize_book_name(match.group(1))
        chapter = int(match.group(2))
        verse = int(match.group(3))
        
        return book_name, chapter, verse
    
    def parse_range_reference(self, reference: str) -> Optional[VerseRange]:
        """
        Parse a Bible reference that might be a range like 'Genesis 1:1-3'
        Returns: VerseRange object or None if not a range
        """
        # Pattern to match: Book Chapter:StartVerse-EndVerse
        range_pattern = r'^(.+?)\s+(\d+):(\d+)-(\d+)$'
        match = re.match(range_pattern, reference.strip())
        
        if match:
            book_name = self.normalize_book_name(match.group(1))
            chapter = int(match.group(2))
            start_verse = int(match.group(3))
            end_verse = int(match.group(4))
            
            # Validate range
            if start_verse > end_verse:
                raise ValueError(f"Invalid range: start verse {start_verse} is greater than end verse {end_verse}")
            
            # Reasonable range limit to prevent abuse
            if end_verse - start_verse > 50:
                raise ValueError(f"Range too large: maximum 50 verses allowed")
            
            return VerseRange(
                book=book_name,
                chapter=chapter,
                start_verse=start_verse,
                end_verse=end_verse
            )
        
        return None
    
    def is_range_reference(self, reference: str) -> bool:
        """Check if the reference is a range (contains dash)"""
        return '-' in reference and self.parse_range_reference(reference) is not None
    
    def chunk_bible_text(self, text: str) -> List[BibleVerseChunk]:
        """
        Parse Bible text and return individual verse chunks
        Expected format: Each line should be "Book Chapter:Verse Text"
        """
        chunks = []
        lines = text.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Pattern to match: "Book Chapter:Verse Text"
                pattern = r'^(.+?)\s+(\d+):(\d+)\s+(.+)$'
                match = re.match(pattern, line)
                
                if match:
                    book_raw = match.group(1)
                    chapter = int(match.group(2))
                    verse = int(match.group(3))
                    text_content = match.group(4)
                    
                    book = self.normalize_book_name(book_raw)
                    reference = f"{book} {chapter}:{verse}"
                    
                    chunk = BibleVerseChunk(
                        book=book,
                        chapter=chapter,
                        verse=verse,
                        text=text_content,
                        reference=reference
                    )
                    chunks.append(chunk)
                else:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    
            except Exception as e:
                print(f"Error parsing line {line_num}: {line} - {str(e)}")
                continue
        
        return chunks
    
    def is_reference_format(self, text: str) -> bool:
        """
        Check if the input text looks like a Bible reference (single, range, or chapter)
        """
        # Enhanced patterns for single references, ranges, and chapters
        reference_patterns = [
            r'^[12]?\s*[A-Za-z]+\.?\s+\d+:\d+$',           # John 3:16
            r'^[12]?\s*[A-Za-z]+\.?\s+\d+:\d+-\d+$',       # John 3:16-18
            r'^[A-Za-z\s]+\s+\d+:\d+$',                    # Song of Solomon 2:1  
            r'^[A-Za-z\s]+\s+\d+:\d+-\d+$',                # Song of Solomon 2:1-3
            r'^[12]?\s*[A-Za-z]+\.?\s+\d+$',               # Genesis 12 (chapter only)
            r'^[A-Za-z\s]+\s+\d+$'                         # Song of Solomon 2 (chapter only)
        ]
        
        for pattern in reference_patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

# import re
# from typing import List, Dict, Tuple, Optional
# from dataclasses import dataclass

# # Define data classes for Bible verse chunks and ranges
# @dataclass
# class BibleVerseChunk:
#     book: str
#     chapter: int
#     verse: int
#     text: str
#     reference: str

# @dataclass
# class VerseRange:
#     book: str
#     chapter: int
#     start_verse: int
#     end_verse: int
    
#     def to_references(self) -> List[str]:
#         """Convert range to list of individual references"""
#         references = []
#         for verse_num in range(self.start_verse, self.end_verse + 1):
#             references.append(f"{self.book} {self.chapter}:{verse_num}")
#         return references

# # Define the BibleChunker class for text chunking and normalization
# class BibleChunker:
#     """
#     Enhanced chunker for Bible text that supports both single verses and ranges
#     """
    
#     def __init__(self):
#         # Bible book mappings for normalization
#         self.book_abbreviations = {
#             # Old Testament
#             "gen": "Genesis", "genesis": "Genesis",
#             "exod": "Exodus", "exodus": "Exodus", "ex": "Exodus",
#             "lev": "Leviticus", "leviticus": "Leviticus",
#             "num": "Numbers", "numbers": "Numbers",
#             "deut": "Deuteronomy", "deuteronomy": "Deuteronomy", "dt": "Deuteronomy",
#             "josh": "Joshua", "joshua": "Joshua",
#             "judg": "Judges", "judges": "Judges",
#             "ruth": "Ruth",
#             "1sam": "1 Samuel", "1 samuel": "1 Samuel", "1 sam": "1 Samuel",
#             "2sam": "2 Samuel", "2 samuel": "2 Samuel", "2 sam": "2 Samuel",
#             "1kgs": "1 Kings", "1 kings": "1 Kings", "1 ki": "1 Kings",
#             "2kgs": "2 Kings", "2 kings": "2 Kings", "2 ki": "2 Kings",
#             "1chr": "1 Chronicles", "1 chronicles": "1 Chronicles", "1 ch": "1 Chronicles",
#             "2chr": "2 Chronicles", "2 chronicles": "2 Chronicles", "2 ch": "2 Chronicles",
#             "ezra": "Ezra",
#             "neh": "Nehemiah", "nehemiah": "Nehemiah",
#             "esth": "Esther", "esther": "Esther",
#             "job": "Job",
#             "ps": "Psalms", "psalm": "Psalms", "psalms": "Psalms", "psa": "Psalms",
#             "prov": "Proverbs", "proverbs": "Proverbs", "pr": "Proverbs",
#             "eccl": "Ecclesiastes", "ecclesiastes": "Ecclesiastes", "ecc": "Ecclesiastes",
#             "song": "Song of Solomon", "songs": "Song of Solomon", "sos": "Song of Solomon",
#             "isa": "Isaiah", "isaiah": "Isaiah", "is": "Isaiah",
#             "jer": "Jeremiah", "jeremiah": "Jeremiah",
#             "lam": "Lamentations", "lamentations": "Lamentations",
#             "ezek": "Ezekiel", "ezekiel": "Ezekiel", "eze": "Ezekiel",
#             "dan": "Daniel", "daniel": "Daniel",
#             "hos": "Hosea", "hosea": "Hosea",
#             "joel": "Joel",
#             "amos": "Amos",
#             "obad": "Obadiah", "obadiah": "Obadiah",
#             "jonah": "Jonah",
#             "mic": "Micah", "micah": "Micah",
#             "nah": "Nahum", "nahum": "Nahum",
#             "hab": "Habakkuk", "habakkuk": "Habakkuk",
#             "zeph": "Zephaniah", "zephaniah": "Zephaniah", "zep": "Zephaniah",
#             "hag": "Haggai", "haggai": "Haggai",
#             "zech": "Zechariah", "zechariah": "Zechariah", "zec": "Zechariah",
#             "mal": "Malachi", "malachi": "Malachi",
            
#             # New Testament
#             "matt": "Matthew", "matthew": "Matthew", "mt": "Matthew",
#             "mark": "Mark", "mk": "Mark",
#             "luke": "Luke", "lk": "Luke",
#             "john": "John", "jn": "John", "joh": "John",
#             "acts": "Acts", "ac": "Acts",
#             "rom": "Romans", "romans": "Romans", "ro": "Romans",
#             "1cor": "1 Corinthians", "1 corinthians": "1 Corinthians", "1 co": "1 Corinthians",
#             "2cor": "2 Corinthians", "2 corinthians": "2 Corinthians", "2 co": "2 Corinthians",
#             "gal": "Galatians", "galatians": "Galatians",
#             "eph": "Ephesians", "ephesians": "Ephesians",
#             "phil": "Philippians", "philippians": "Philippians", "php": "Philippians",
#             "col": "Colossians", "colossians": "Colossians",
#             "1thess": "1 Thessalonians", "1 thessalonians": "1 Thessalonians", "1 th": "1 Thessalonians",
#             "2thess": "2 Thessalonians", "2 thessalonians": "2 Thessalonians", "2 th": "2 Thessalonians",
#             "1tim": "1 Timothy", "1 timothy": "1 Timothy", "1 ti": "1 Timothy",
#             "2tim": "2 Timothy", "2 timothy": "2 Timothy", "2 ti": "2 Timothy",
#             "titus": "Titus", "tit": "Titus",
#             "phlm": "Philemon", "philemon": "Philemon", "phm": "Philemon",
#             "heb": "Hebrews", "hebrews": "Hebrews",
#             "jas": "James", "james": "James", "ja": "James",
#             "1pet": "1 Peter", "1 peter": "1 Peter", "1 pe": "1 Peter",
#             "2pet": "2 Peter", "2 peter": "2 Peter", "2 pe": "2 Peter",
#             "1john": "1 John", "1 jn": "1 John", "1 jo": "1 John",
#             "2john": "2 John", "2 jn": "2 John", "2 jo": "2 John",
#             "3john": "3 John", "3 jn": "3 John", "3 jo": "3 John",
#             "jude": "Jude",
#             "rev": "Revelation", "revelation": "Revelation", "re": "Revelation"
#         }
    
#     def normalize_book_name(self, book: str) -> str:
#         """Normalize book name to standard format"""
#         book_lower = book.lower().strip()
#         return self.book_abbreviations.get(book_lower, book.title())
    
#     def parse_reference(self, reference: str) -> Tuple[str, int, int]:
#         """
#         Parse a single Bible reference like 'John 3:16' into components
#         Returns: (book, chapter, verse)
#         """
#         # Pattern to match: Book Chapter:Verse
#         pattern = r'^(.+?)\s+(\d+):(\d+)$'
#         match = re.match(pattern, reference.strip())
        
#         if not match:
#             raise ValueError(f"Invalid Bible reference format: {reference}")
        
#         book_name = self.normalize_book_name(match.group(1))
#         chapter = int(match.group(2))
#         verse = int(match.group(3))
        
#         return book_name, chapter, verse
    
#     def parse_range_reference(self, reference: str) -> Optional[VerseRange]:
#         """
#         Parse a Bible reference that might be a range like 'Genesis 1:1-3'
#         Returns: VerseRange object or None if not a range
#         """
#         # Pattern to match: Book Chapter:StartVerse-EndVerse
#         range_pattern = r'^(.+?)\s+(\d+):(\d+)-(\d+)$'
#         match = re.match(range_pattern, reference.strip())
        
#         if match:
#             book_name = self.normalize_book_name(match.group(1))
#             chapter = int(match.group(2))
#             start_verse = int(match.group(3))
#             end_verse = int(match.group(4))
            
#             # Validate range
#             if start_verse > end_verse:
#                 raise ValueError(f"Invalid range: start verse {start_verse} is greater than end verse {end_verse}")
            
#             # Reasonable range limit to prevent abuse
#             if end_verse - start_verse > 50:
#                 raise ValueError(f"Range too large: maximum 50 verses allowed")
            
#             return VerseRange(
#                 book=book_name,
#                 chapter=chapter,
#                 start_verse=start_verse,
#                 end_verse=end_verse
#             )
        
#         return None
    
#     def is_range_reference(self, reference: str) -> bool:
#         """Check if the reference is a range (contains dash)"""
#         return '-' in reference and self.parse_range_reference(reference) is not None
    
#     def chunk_bible_text(self, text: str) -> List[BibleVerseChunk]:
#         """
#         Parse Bible text and return individual verse chunks
#         Expected format: Each line should be "Book Chapter:Verse Text"
#         """
#         chunks = []
#         lines = text.strip().split('\n')
        
#         for line_num, line in enumerate(lines, 1):
#             line = line.strip()
#             if not line:
#                 continue
                
#             try:
#                 # Pattern to match: "Book Chapter:Verse Text"
#                 pattern = r'^(.+?)\s+(\d+):(\d+)\s+(.+)$'
#                 match = re.match(pattern, line)
                
#                 if match:
#                     book_raw = match.group(1)
#                     chapter = int(match.group(2))
#                     verse = int(match.group(3))
#                     text_content = match.group(4)
                    
#                     book = self.normalize_book_name(book_raw)
#                     reference = f"{book} {chapter}:{verse}"
                    
#                     chunk = BibleVerseChunk(
#                         book=book,
#                         chapter=chapter,
#                         verse=verse,
#                         text=text_content,
#                         reference=reference
#                     )
#                     chunks.append(chunk)
#                 else:
#                     print(f"Warning: Could not parse line {line_num}: {line}")
                    
#             except Exception as e:
#                 print(f"Error parsing line {line_num}: {line} - {str(e)}")
#                 continue
        
#         return chunks
    
#     def is_reference_format(self, text: str) -> bool:
#         """
#         Check if the input text looks like a Bible reference (single or range)
#         """
#         # Enhanced patterns for both single references and ranges
#         reference_patterns = [
#             r'^[12]?\s*[A-Za-z]+\.?\s+\d+:\d+$',           # John 3:16
#             r'^[12]?\s*[A-Za-z]+\.?\s+\d+:\d+-\d+$',       # John 3:16-18
#             r'^[A-Za-z\s]+\s+\d+:\d+$',                    # Song of Solomon 2:1  
#             r'^[A-Za-z\s]+\s+\d+:\d+-\d+$'                 # Song of Solomon 2:1-3
#         ]
        
#         for pattern in reference_patterns:
#             if re.match(pattern, text.strip()):
#                 return True
#         return False
