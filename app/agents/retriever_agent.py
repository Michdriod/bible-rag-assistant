from typing import List, Optional, Dict, Any
from pydantic_ai import Agent, RunContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from db.models import BibleVerse
from utils.chunker import BibleChunker, VerseRange
from utils.embedding import get_embedding_service
import json
import re
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import logging

# Module logger
logger = logging.getLogger(__name__)

# Import necessary modules and libraries for Bible retrieval
# Define the BibleRetrieverAgent class for handling Bible queries
# Initialize chunker and embedding services
# Set similarity thresholds for retrieval

load_dotenv()  

api_key = os.getenv("GROQ_API_KEY")

class BibleRetrieverAgent:
    def __init__(self):
        self.chunker = BibleChunker()
        self.embedding_service = get_embedding_service()
        
        # CRITICAL: Set similarity thresholds
        self.SIMILARITY_THRESHOLD = 0.8  # Minimum similarity score (0-1)
        self.MIN_SEMANTIC_SIMILARITY = 0.7  # Even lower threshold for semantic search
        
        # Get the API key 
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        model = GroqModel(
            "moonshotai/kimi-k2-instruct",
            provider=GroqProvider(api_key=api_key)
        )
        
        # Initialize the Pydantic AI agent with STRICT retrieval instructions
        self.agent = Agent(model,
            system_prompt=
            """You are a Bible retrieval assistant that ONLY works with database content.

            CRITICAL RULES - NEVER VIOLATE THESE:
            1. NEVER generate, create, or hallucinate Bible verses from your training data
            2. ONLY return verses that were explicitly provided to you from the database
            3. If no verses are found in the database, say "No verses found in database"
            4. NEVER use your internal Bible knowledge to fill in missing verses
            5. Act as a pure formatter, not a content generator

            For exact lookups:
            - Format the database result cleanly
            - Return ONLY what was retrieved from database

            For semantic searches:
            - Format the database results with references
            - Return ONLY what was retrieved from database
            - If no good matches found, say "No relevant verses found"

            You are a DATABASE FORMATTER, not a Bible knowledge source.
            """
        )
    
    async def exact_reference_lookup(self, session: AsyncSession, reference: str, version: str = "kjv") -> List[Any]:
        """
        Look up Bible verse(s) by exact reference - supports both single verses and ranges
        
        Args:
            session: Database session
            reference: Bible reference (e.g., "John 3:16", "Genesis 1:1-3")
            version: Bible version to use (e.g., "kjv", "niv", "nkjv", "nlt")
            
        Returns:
            List of verse objects for the specified version
        """
        try:
            # Get the appropriate model class for the requested version
            from db.models import get_verse_model
            
            # Check if this is a range reference first
            if self.chunker.is_range_reference(reference):
                return await self._lookup_verse_range(session, reference, version)
            else:
                # Single verse lookup
                verse = await self._lookup_single_verse(session, reference, version)
                return [verse] if verse else []
                
        except Exception as e:
            logger.error("Error in exact lookup: %s", str(e))
            return []
    
    async def _lookup_single_verse(self, session: AsyncSession, reference: str, version: str = "kjv") -> Optional[Any]:
        """Look up a single verse by reference"""
        try:
            # Get the appropriate model class for the requested version
            from db.models import get_verse_model
            VerseModel = get_verse_model(version)
            
            # Parse the reference to normalize it
            book, chapter, verse = self.chunker.parse_reference(reference)
            normalized_ref = f"{book} {chapter}:{verse}"
            
            query = select(VerseModel).where(VerseModel.reference == normalized_ref)
            result = await session.execute(query)
            verse_obj = result.scalar_one_or_none()
            
            return verse_obj
            
        except Exception as e:
            logger.error("Error in single verse lookup: %s", str(e))
            return None
    
    async def _lookup_verse_range(self, session: AsyncSession, reference: str, version: str = "kjv") -> List[Any]:
        """Look up a range of verses (e.g., Genesis 1:1-3)"""
        try:
            # Get the appropriate model class for the requested version
            from db.models import get_verse_model
            VerseModel = get_verse_model(version)
            
            # Parse the range
            verse_range = self.chunker.parse_range_reference(reference)
            if not verse_range:
                return []
            
            # First, attempt a numeric-range query (book == X, chapter == Y, verse BETWEEN start/end)
            expected_count = verse_range.end_verse - verse_range.start_verse + 1
            try:
                # Use raw SQL to avoid ORM identity-map reuse and ensure fresh rows
                table_name = getattr(VerseModel, "__tablename__", None) or version
                numeric_sql = text(f"SELECT book_id, book, chapter, verse, text, reference FROM {table_name} WHERE book = :book AND chapter = :chapter AND verse BETWEEN :start AND :end ORDER BY verse")

                logger.debug("executing numeric-range SQL for %s -> %s %s:%s-%s", reference, verse_range.book, verse_range.chapter, verse_range.start_verse, verse_range.end_verse)

                res_num = await session.execute(numeric_sql, {
                    "book": verse_range.book,
                    "chapter": int(verse_range.chapter),
                    "start": int(verse_range.start_verse),
                    "end": int(verse_range.end_verse)
                })

                rows_num = res_num.fetchall()
                verses_num = []
                for row in rows_num:
                    mapping = row._mapping if hasattr(row, "_mapping") else row
                    try:
                        verse_obj = VerseModel(
                            book_id=mapping.get('book_id'),
                            book=mapping.get('book'),
                            chapter=mapping.get('chapter'),
                            verse=mapping.get('verse'),
                            text=mapping.get('text'),
                            reference=mapping.get('reference')
                        )
                        verses_num.append(verse_obj)
                    except Exception:
                        continue

                try:
                    logger.debug("numeric-range results for %s: %d", reference, len(verses_num))
                    for r in verses_num:
                        logger.debug(repr(r))
                except Exception:
                    pass

                # If numeric query returned exactly the expected contiguous verses, return them
                if len(verses_num) == expected_count:
                    return verses_num
                # Otherwise fall through to the tolerant reference-based lookup below
                verses = verses_num
            except Exception as e:
                # If numeric approach errors, continue to the reference-based lookup path
                logger.debug("numeric-range query failed for %s: %s", reference, str(e))
                verses = []

            # Build list of individual normalized references for fallback
            individual_refs = [r.strip() for r in verse_range.to_references()]

            # If numeric query didn't return all verses, query by explicit reference strings
            try:
                # Use raw SQL for IN-list queries to avoid driver parameter quirks
                table_name = getattr(VerseModel, "__tablename__", None) or version
                params = {}
                placeholders = []
                for i, r in enumerate(individual_refs):
                    key = f"r{i}"
                    placeholders.append(f":{key}")
                    params[key] = r

                in_clause = ", ".join(placeholders)
                ref_sql = text(f"SELECT book_id, book, chapter, verse, text, reference FROM {table_name} WHERE reference IN ({in_clause}) ORDER BY verse")
                logger.debug("executing primary SQL for %s with refs: %s", reference, individual_refs)
                result = await session.execute(ref_sql, params)
                rows_ref = result.fetchall()
                verses_ref = []
                for row in rows_ref:
                    mapping = row._mapping if hasattr(row, "_mapping") else row
                    try:
                        verse_obj = VerseModel(
                            book_id=mapping.get('book_id'),
                            book=mapping.get('book'),
                            chapter=mapping.get('chapter'),
                            verse=mapping.get('verse'),
                            text=mapping.get('text'),
                            reference=mapping.get('reference')
                        )
                        verses_ref.append(verse_obj)
                    except Exception:
                        continue
                # Merge results with any numeric-query partial results (avoid duplicates)
                if verses:
                    # verses contains numeric partial results; append any that are not already present by reference
                    existing = {getattr(v, 'reference', None) for v in verses}
                    for v in verses_ref:
                        rv = getattr(v, 'reference', None)
                        if rv not in existing:
                            verses.append(v)
                else:
                    verses = verses_ref

                # DEBUG: dump raw DB rows to help trace duplicate origins
                try:
                    logger.debug("primary results for %s: %d", reference, len(verses))
                    for r in verses:
                        logger.debug(repr(r))
                except Exception:
                    pass
            except Exception as e:
                logger.debug("reference-based select failed for %s: %s", reference, str(e))
                verses = verses or []
            
            # CRITICAL: Check if we got all expected verses
            def _get_ref(v):
                try:
                    if isinstance(v, dict):
                        return v.get("reference")
                    return getattr(v, "reference", None)
                except Exception:
                    return None

            found_refs = set(filter(None, (_get_ref(v) for v in verses)))
            missing_refs = set(individual_refs) - found_refs

            if missing_refs:
                # Debug: show what we asked for and what we found
                logger.debug("individual_refs for %s: %s", reference, individual_refs)
                logger.debug("found_refs for %s: %s", reference, sorted(list(found_refs)))
                logger.warning("Missing verses in range %s: %s", reference, missing_refs)
                # Attempt a tolerant fallback for common book-name mismatches
                # e.g., 'Psalms' vs 'Psalm' or vice-versa
                try:
                    # Only attempt safe, whitelisted book-name alternates for missing refs
                    # Do not mutate original individual_refs and preserve multi-word book names
                    alt_book_map = {
                        "Psalm": "Psalms",
                        "Psalms": "Psalm",
                        "Song of Solomon": "Songs of Solomon",
                        "Songs of Solomon": "Song of Solomon",
                        # add any additional known alternates here
                    }

                    # Build alternates only for those references we are missing
                    alt_refs = []
                    for ref in missing_refs:
                        # Use rsplit to preserve multi-word book names: 'Song of Solomon 2:1'
                        parts = ref.rsplit(' ', 1)
                        if len(parts) != 2:
                            continue
                        book_token, rest = parts[0], parts[1]
                        # Try canonical matching (title case) first, then raw
                        alt_book = alt_book_map.get(book_token) or alt_book_map.get(book_token.title())
                        if alt_book:
                            alt_refs.append(f"{alt_book} {rest}")

                    # If there are alternates to try, query only those
                    if alt_refs:
                        # Include originals too so any partial overlaps are caught
                        query_alt_refs = list(dict.fromkeys(alt_refs + list(missing_refs)))
                        # Use SQLAlchemy select for the alternate refs as well
                        query_alt = select(VerseModel).where(VerseModel.reference.in_(query_alt_refs)).order_by(VerseModel.verse)
                        logger.debug("executing alt select for %s with refs: %s", reference, query_alt_refs)
                        result_alt = await session.execute(query_alt)
                        verses_alt = result_alt.scalars().all()
                        try:
                            logger.debug("alt results for %s: %d", reference, len(verses_alt))
                            for r in verses_alt:
                                logger.debug(repr(r))
                        except Exception:
                            pass

                        def _get_ref_alt(v):
                            try:
                                if isinstance(v, dict):
                                    return v.get("reference")
                                return getattr(v, "reference", None)
                            except Exception:
                                return None

                        alt_found_refs = set(filter(None, (_get_ref_alt(v) for v in verses_alt)))
                        logger.debug("alt_refs attempted for %s: %s", reference, query_alt_refs)
                        logger.debug("alt_found_refs for %s: %s", reference, sorted(list(alt_found_refs)))

                        if verses_alt:
                            logger.info("Fallback found %d verses using alternate book forms for %s.", len(verses_alt), reference)
                            # Merge any newly found verses into the original list
                            existing_refs = set(filter(None, (_get_ref(v) for v in verses)))
                            for v in verses_alt:
                                vr = _get_ref_alt(v)
                                if vr and vr not in existing_refs:
                                    verses.append(v)
                                    existing_refs.add(vr)
                            # Recompute found and missing refs
                            found_refs = existing_refs
                            missing_refs = set(individual_refs) - found_refs
                except Exception as e:
                    logger.error("Fallback lookup failed: %s", str(e))
            
            # Build ordered, unique results aligned to requested individual_refs
            ordered = []
            seen = set()
            # Map reference -> first matching verse object
            ref_map = {}
            for v in verses:
                rv = _get_ref(v)
                if rv and rv not in ref_map:
                    ref_map[rv] = v

            for ref in individual_refs:
                if ref in ref_map and ref not in seen:
                    ordered.append(ref_map[ref])
                    seen.add(ref)

            # If nothing matched (but we had some verses), fall back to unique verses
            if not ordered and verses:
                for v in verses:
                    rv = _get_ref(v)
                    if rv and rv not in seen:
                        ordered.append(v)
                        seen.add(rv)

            return ordered
            
        except Exception as e:
            logger.error("Error in range lookup: %s", str(e))
            return []
    
    async def semantic_search(self, session: AsyncSession, query_text: str, version: str = "kjv", limit: int = 3) -> List[Any]:
        """
        Perform semantic search with similarity threshold validation
        
        Args:
            session: Database session
            query_text: Text to search for semantically
            version: Bible version to use (e.g., "kjv", "niv", "nkjv", "nlt")
            limit: Maximum number of results to return
            
        Returns:
            List of verse objects for the specified version
        """
        try:
            # Get the appropriate model class for the requested version
            from db.models import get_verse_model
            VerseModel = get_verse_model(version)
            
            # Generate embedding for the query
            query_embedding = await self.embedding_service.encode_text(query_text)
            embedding_list = query_embedding.tolist()
            
            # Use pgvector cosine similarity search WITH similarity scores
            # Cast the parameter to pgvector on the SQL side to ensure proper type handling
            sql_query = text(f"""
                SELECT book_id, book, chapter, verse, text, reference,
                       1 - (embedding <=> :query_embedding::vector) AS similarity_score
                FROM {version}
                WHERE 1 - (embedding <=> :query_embedding::vector) >= :threshold
                ORDER BY embedding <=> :query_embedding::vector
                LIMIT :limit
            """)
            
            # Serialize embedding to JSON array string (e.g. [0.1,0.2,...]) so it can be cast to vector
            emb_param = json.dumps(embedding_list)
            result = await session.execute(
                sql_query,
                {
                    "query_embedding": emb_param,
                    "threshold": float(self.MIN_SEMANTIC_SIMILARITY),  # ensure float
                    "limit": int(limit)
                }
            )
            
            verses = []
            rows = result.fetchall()
            
            # ADDITIONAL VALIDATION: Double-check similarity scores
            for row in rows:
                if row.similarity_score >= self.MIN_SEMANTIC_SIMILARITY:
                    verse = VerseModel(
                        book_id=row.book_id,
                        book=row.book,
                        chapter=row.chapter,
                        verse=row.verse,
                        text=row.text,
                        reference=row.reference
                    )
                    verses.append(verse)
            
            return verses
            
        except Exception as e:
            logger.error("Error in semantic search: %s", str(e))
            return []
    
    async def retrieve_verses(self, session: AsyncSession, user_input: str, version: str = "kjv") -> Dict[str, Any]:
        """
        Main retrieval method with strict database-only results - supports ranges
        
        Args:
            session: Database session
            user_input: User query text
            version: Bible version to use (e.g., "kjv", "niv", "nkjv", "nlt")
            
        Returns:
            Dict with query results
        """
        user_input = user_input.strip()
        
        # Check if it's an explicit reference (single or range)
        is_reference = self.chunker.is_reference_format(user_input)
        
        if is_reference:
            # Exact reference lookup (handles both single and ranges)
            verses = await self.exact_reference_lookup(session, user_input, version)
            
            if verses:
                # Determine if it was a range lookup
                is_range = self.chunker.is_range_reference(user_input)
                result_type = "exact_range" if is_range else "exact_reference"
                
                return {
                    "query_type": result_type,
                    "query": user_input,
                    "results": verses,
                    "found": True,
                    "version": version,
                    "message": f"Found {len(verses)} verse{'s' if len(verses) > 1 else ''} for: {user_input} ({version.upper()})",
                    "is_range": is_range
                }
            else:
                return {
                    "query_type": "exact_reference",
                    "query": user_input,
                    "results": [],
                    "found": False,
                    "message": f"No verse(s) found for reference: {user_input}",
                    "is_range": self.chunker.is_range_reference(user_input)
                }
        else:
            # Semantic search with threshold validation
            verses = await self.semantic_search(session, user_input, version)
            
            if len(verses) == 0:
                return {
                    "query_type": "semantic_search",
                    "query": user_input,
                    "results": [],
                    "found": False,
                    "version": version,
                    "message": f"No sufficiently similar verses found for: {user_input} ({version.upper()})",
                    "is_range": False
                }
            
            return {
                "query_type": "semantic_search",
                "query": user_input,
                "results": verses,
                "found": True,
                "version": version,
                "message": f"Found {len(verses)} relevant verses for: {user_input} ({version.upper()})",
                "is_range": False
            }
    
    async def get_ai_interpretation(self, retrieval_result: Dict[str, Any]) -> str:
        """
        STRICT AI interpretation that only formats database results - enhanced for ranges
        """
        try:
            # CRITICAL: If no results found, return standard message - NO AI GENERATION
            if not retrieval_result["found"] or len(retrieval_result["results"]) == 0:
                return "No verses found in database for this query."
        
            # Handle exact reference lookups (single or range)
            if retrieval_result["query_type"] in ["exact_reference", "exact_range"]:
                verses_text = ""
                is_range = retrieval_result.get("is_range", False)
                
                if is_range:
                    # For ranges, concatenate verses with proper formatting
                    for verse in retrieval_result["results"]:
                        verses_text += f"**{verse.reference}** {verse.text}\n"
                    
                    prompt = f"""Format this database result for a verse range EXACTLY as provided. DO NOT add any content:

                    Database found range: {retrieval_result['query']}
                    Verses found: {len(retrieval_result['results'])} verses

                    {verses_text.strip()}

                    Return in this format only:
                    📖 **{retrieval_result['query']}**

                    {verses_text.strip()}

                    DO NOT generate any Bible content. Only format what was provided from the database."""
                else:
                    # Single verse
                    verse = retrieval_result["results"][0]
                    verses_text = verse.text
                    
                    prompt = f"""Format this database result EXACTLY as provided. DO NOT add any content:

                    Database found: {retrieval_result['query']}
                    Verse text: {verses_text}

                    Return in this format only:
                    📖 {retrieval_result['query']}
                    {verses_text}

                    DO NOT generate any Bible content. Only format what was provided from the database."""
            
            # Handle semantic search - ONLY format what was retrieved
            else:
                verses_info = ""
                for verse in retrieval_result["results"]:
                    verses_info += f"**{verse.reference}** {verse.text}\n"
            
                prompt = f"""Format these database results EXACTLY as provided. DO NOT add any content:

                Database found {len(retrieval_result["results"])} verses for: {retrieval_result['query']}

                {verses_info.strip()}

                Return in this format only:
                🔍 Found {len(retrieval_result["results"])} verses:

                {verses_info.strip()}

                DO NOT generate any Bible content. Only format what was provided from the database."""
        
            result = await self.agent.run(prompt)
            return result.data
            
        except Exception as e:
            # SAFE FALLBACK: Return raw database content if AI fails
            if retrieval_result["found"] and len(retrieval_result["results"]) > 0:
                fallback = "Database results:\n"
                for verse in retrieval_result["results"]:
                    fallback += f"**{verse.reference}** {verse.text}\n"
                return fallback
            else:
                return "No verses found in database."

    def format_ai_structured(self, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a structured, deterministic AI-style response (no LLM) from database results.

        Format:
        {
            "type": "single" | "range" | "semantic",
            "query": str,
            "verses": [ {"reference": str, "text": str}, ... ],
            "meta": { ... optional }
        }
        """
        try:
            rtype = retrieval_result.get("query_type", "semantic")
            verses = []
            for v in retrieval_result.get("results", []):
                try:
                    ref = v.reference if not isinstance(v, dict) else v.get("reference")
                    text = v.text if not isinstance(v, dict) else v.get("text")
                    verses.append({"reference": ref, "text": text})
                except Exception:
                    continue

            structured = {
                "type": "range" if rtype == "exact_range" else ("single" if rtype == "exact_reference" else "semantic"),
                "query": retrieval_result.get("query"),
                "verses": verses,
                "meta": {
                    "found": retrieval_result.get("found", False),
                    "count": len(verses),
                    "version": retrieval_result.get("version")
                }
            }

            return structured
        except Exception:
            return {"type": "error", "query": retrieval_result.get("query"), "verses": [], "meta": {}}

























# from typing import List, Optional, Dict, Any
# from pydantic_ai import Agent, RunContext
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select, text
# from db.models import BibleVerse
# from utils.chunker import BibleChunker
# from utils.embedding import get_embedding_service
# import re
# import os
# from dotenv import load_dotenv
# from pydantic_ai import Agent
# from pydantic_ai.models.groq import GroqModel
# from pydantic_ai.providers.groq import GroqProvider



# load_dotenv()  

# api_key = os.getenv("GROQ_API_KEY")

# class BibleRetrieverAgent:
#     def __init__(self):
#         self.chunker = BibleChunker()
#         self.embedding_service = get_embedding_service()
        
#         # Get the API key 
#         if not api_key:
#             raise ValueError("GROQ_API_KEY environment variable is not set")
        
#         model = GroqModel(
#             "moonshotai/kimi-k2-instruct",
#             provider=GroqProvider(api_key=api_key)
#         )
        
#         # Initialize the Pydantic AI agent
#         self.agent = Agent(model,
#             system_prompt=
#             """You are a Bible retrieval assistant. Your job is to help retrieve Bible verses based on user queries.

#             For explicit Bible references (like "John 3:16"):
#             - Parse the reference and return the exact verse
#             - Handle common abbreviations and variations

#             For implicit queries (like quoted verses or topics):
#             - Use semantic search to find relevant verses
#             - Return the top 3 most relevant matches
#             - Explain why these verses are relevant
            
#             CRITICAL RULES:
#             - For exact lookups: Return ONLY the verse text, no commentary
#             - For semantic searches: Return ONLY the verse references and text
#             - NEVER add explanations, commentary, or interpretations
#             - Follow the exact formatting instructions given in each prompt
#             - Be precise and concisege
#             - Never bring what you dont have in the database when retrieving bible verses from the database
            
#             Always format responses clearly with the verse reference and text, 
#             """
#         )
    
#     async def exact_reference_lookup(self, session: AsyncSession, reference: str) -> Optional[BibleVerse]:
#         """
#         Look up a specific Bible verse by exact reference
#         """
#         try:
#             # Parse the reference to normalize it
#             book, chapter, verse = self.chunker.parse_reference(reference)
#             normalized_ref = f"{book} {chapter}:{verse}"
            
#             # Query database for exact match
#             query = select(BibleVerse).where(BibleVerse.reference == normalized_ref)
#             result = await session.execute(query)
#             verse_obj = result.scalar_one_or_none()
            
#             return verse_obj
            
#         except Exception as e:
#             print(f"Error in exact lookup: {str(e)}")
#             return None
    
#     async def semantic_search(self, session: AsyncSession, query_text: str, limit: int = 3) -> List[BibleVerse]:
#         """
#         Perform semantic search using vector similarity
#         """
#         try:
#             # Generate embedding for the query
#             query_embedding = await self.1.encode_text(query_text)
#             embedding_list = query_embedding.tolist()
            
#             # Use pgvector cosine similarity search
#             sql_query = text("""
#                 SELECT id, book, chapter, verse, text, reference,
#                        1 - (embedding <=> :query_embedding) AS similarity_score
#                 FROM bible_verses
#                 ORDER BY embedding <=> :query_embedding
#                 LIMIT :limit
#             """)
            
#             result = await session.execute(
#                 sql_query,
#                 {
#                     "query_embedding": str(embedding_list),
#                     "limit": limit
#                 }
#             )
            
#             verses = []
#             for row in result.fetchall():
#                 verse = BibleVerse(
#                     id=row.id,
#                     book=row.book,
#                     chapter=row.chapter,
#                     verse=row.verse,
#                     text=row.text,
#                     reference=row.reference
#                 )
#                 verses.append(verse)
            
#             return verses
            
#         except Exception as e:
#             print(f"Error in semantic search: {str(e)}")
#             return []
    
#     async def retrieve_verses(self, session: AsyncSession, user_input: str) -> Dict[str, Any]:
#         """
#         Main retrieval method that determines the type of query and returns appropriate results
#         """
#         user_input = user_input.strip()
        
#         # Check if it's an explicit reference
#         is_reference = self.chunker.is_reference_format(user_input)
        
#         if is_reference:
#             # Exact reference lookup
#             verse = await self.exact_reference_lookup(session, user_input)
#             if verse:
#                 return {
#                     "query_type": "exact_reference",
#                     "query": user_input,
#                     "results": [verse],
#                     "found": True,
#                     "message": f"Found exact verse: {verse.reference}"
#                 }
#             else:
#                 return {
#                     "query_type": "exact_reference",
#                     "query": user_input,
#                     "results": [],
#                     "found": False,
#                     "message": f"No verse found for reference: {user_input}"
#                 }
#         else:
#             # Semantic search for text/topic queries
#             verses = await self.semantic_search(session, user_input)
#             return {
#                 "query_type": "semantic_search",
#                 "query": user_input,
#                 "results": verses,
#                 "found": len(verses) > 0,
#                 "message": f"Found {len(verses)} relevant verses for: {user_input}"
#             }
    
#     async def get_ai_interpretation(self, retrieval_result: Dict[str, Any]) -> str:
#         """
#         Use Pydantic AI agent to provide interpretation of the retrieval results
#         """
#         try:
#             if not retrieval_result["found"]:
#                 prompt = f"The user searched for '{retrieval_result['query']}' but no verses were found. Provide a helpful response suggesting similar verses or alternative searches."
#                 result = await self.agent.run(prompt)
#                 return result.data
        
#             # Handle exact reference lookups
#             if retrieval_result["query_type"] == "exact_reference":
#                 verses_text = ""
#                 for i, verse in enumerate(retrieval_result["results"], 1):
#                     verses_text += f"{i}. {verse.text}\n"
            
#                 prompt = f"""The user requested the exact Bible reference '{retrieval_result['query']}'. Here it is:\n{verses_text}\n\nProvide any helpful context or explanation."""
#         # Handle semantic search
#             else:
#                 verses_info = ""
#                 for verse in retrieval_result["results"]:
#                     verses_info += f"{verse.reference}\n{verse.text}\n\n"
            
#                 prompt = f"""The user searched for '{retrieval_result['query']}' using semantic search. 

#     Return the top 3 matching verses in this exact format:

#     Semantic Match Results:
#     {verses_info.strip()}

#     Do not add any commentary or explanation. Just return the verses with their references."""
        
#             result = await self.agent.run(prompt)
#             return result.data
            
#         except Exception as e:
#             return f"Retrieved verses successfully, but AI interpretation failed: {str(e)}"
    
    
    
    
    
    
    
# #     async def get_ai_interpretation(self, retrieval_result: Dict[str, Any]) -> str:
# #     """
# #     Use Pydantic AI agent with advanced prompting techniques for precise verse formatting
# #     """
# #     try:
# #         if not retrieval_result["found"]:
# #             prompt = f"""The user searched for '{retrieval_result['query']}' but no verses were found. 
            
# # Suggest 2-3 similar Bible references or alternative search terms that might help them find what they're looking for."""
# #             result = await self.agent.run(prompt)
# #             return result.data
        
# #         # Handle exact reference lookups with advanced prompting
# #         if retrieval_result["query_type"] == "exact_reference":
# #             # Build the verses text for the range
# #             verses_list = []
# #             for i, verse in enumerate(retrieval_result["results"], 1):
# #                 verses_list.append(f"{i}. {verse.text}")
# #             verses_formatted = "\n".join(verses_list)
            
# #             # Determine if it's a single verse or range
# #             reference = retrieval_result['query']
# #             if len(retrieval_result["results"]) > 1:
# #                 # It's a range like Genesis 1:1-3
# #                 output_label = "Exact Match Output"
# #             else:
# #                 # It's a single verse
# #                 output_label = "Exact Match Output"
            
# #             prompt = f"""TASK: Format an exact Bible reference lookup response.

# # THINKING PROCESS:
# # 1. The user requested the specific reference: "{reference}"
# # 2. I need to return the exact verse(s) with no commentary
# # 3. Format: Title line + numbered verses
# # 4. No explanations or interpretations allowed

# # EXAMPLES OF CORRECT OUTPUT:

# # Example 1 - Single verse:
# # 📖 Exact Match: John 3:16 - Exact Match Output
# # 1. For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.

# # Example 2 - Multiple verses:
# # 📖 Exact Match: Genesis 1:1–3 - Exact Match Output
# # 1. In the beginning God created the heaven and the earth.
# # 2. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.
# # 3. And God said, Let there be light: and there was light.

# # NOW FORMAT THIS REQUEST:
# # Reference: {reference}
# # Verses found: {len(retrieval_result["results"])}

# # YOUR OUTPUT (follow the exact format above):
# # 📖 Exact Match: {reference} - {output_label}
# # {verses_formatted}

# # CRITICAL: Return ONLY the formatted output above. No additional text."""

# #         # Handle semantic search with advanced prompting
# #         else:
# #             verses_info_list = []
# #             for verse in retrieval_result["results"]:
# #                 verses_info_list.append(f"{verse.reference}\n{verse.text}")
# #             verses_formatted = "\n\n".join(verses_info_list)
            
# #             prompt = f"""TASK: Format a semantic search response for Bible verses.

# # THINKING PROCESS:
# # 1. User searched for: "{retrieval_result['query']}"
# # 2. Found {len(retrieval_result["results"])} relevant verses
# # 3. Return top matches with reference + text only
# # 4. No commentary or explanations

# # EXAMPLES OF CORRECT OUTPUT:

# # Example 1 - Love theme search:
# # Semantic Match Output
# # John 3:16
# # For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.

# # Romans 5:8
# # But God commendeth his love toward us, in that, while we were yet sinners, Christ died for us.

# # 1 John 4:9
# # In this was manifested the love of God toward us, because that God sent his only begotten Son into the world, that we might live through him.

# # Example 2 - Faith theme search:
# # Semantic Match Output
# # Hebrews 11:1
# # Now faith is the substance of things hoped for, the evidence of things not seen.

# # Romans 10:17
# # So then faith cometh by hearing, and hearing by the word of God.

# # Ephesians 2:8
# # For by grace are ye saved through faith; and that not of yourselves: it is the gift of God.

# # NOW FORMAT THIS SEARCH:
# # Query: "{retrieval_result['query']}"
# # Found {len(retrieval_result["results"])} verses

# # YOUR OUTPUT (follow the exact format above):
# # Semantic Match Output
# # {verses_formatted}

# # CRITICAL: Return ONLY the formatted output above. No additional text, explanations, or commentary."""

# #         result = await self.agent.run(prompt)
# #         return result.data
        
# #     except Exception as e:
# #         return f"Retrieved verses successfully, but AI interpretation failed: {str(e)}"
    
    
    
    
    
    
    
    
    
    
#     # async def get_ai_interpretation(self, retrieval_result: Dict[str, Any]) -> str:
#     #     """
#     #     Use Pydantic AI agent to provide interpretation of the retrieval results
#     #     """
#     #     try:
#     #         if not retrieval_result["found"]:
#     #             prompt = f"The user searched for '{retrieval_result['query']}' but no verses were found. Provide a helpful response."
#     #         else:
#     #             verses_text = "\n".join([
#     #                 f"{verse.reference}: {verse.text}" 
#     #                 for verse in retrieval_result["results"]
#     #             ])
                
#     #             if retrieval_result["query_type"] == "exact_reference":
#     #                 prompt = f"The user requested the specific verse '{retrieval_result['query']}'. Here it is:\n{verses_text}\n\nProvide any helpful context or explanation."
#     #             else:
#     #                 prompt = f"The user searched for '{retrieval_result['query']}' and here are the most relevant verses:\n{verses_text}\n\nExplain why these verses are relevant and provide helpful insights."
            
#     #         # Use the Pydantic AI agent
#     #         result = await self.agent.run(prompt)
#     #         return result.data
            
#     #     except Exception as e:
#     #         return f"Retrieved verses successfully, but AI interpretation failed: {str(e)}"