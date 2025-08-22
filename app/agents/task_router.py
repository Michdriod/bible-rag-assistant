from typing import Dict, Any, List
import re
from pydantic_ai import Agent
from sqlalchemy.ext.asyncio import AsyncSession
from app.agents.retriever_agent import BibleRetrieverAgent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider      
from utils.chunker import BibleChunker
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Import necessary modules and libraries for task routing
# Define the TaskRouterAgent class for query classification and routing
# Initialize retriever agent and chunker
# Set up the routing agent with strict instructions

load_dotenv()  

api_key = os.getenv("GROQ_API_KEY")

class TaskRouterAgent:
    """
    Main routing agent that determines the appropriate action based on user input
    and coordinates with specialized agents with strict database-only responses
    """
    
    def __init__(self):
        self.retriever_agent = BibleRetrieverAgent()
        self.chunker = BibleChunker()
    
        # Get the API key
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable is not set - LLM-based features will be disabled")
            self.router_agent = None
        else:
            model = GroqModel(
                "llama-3.1-8b-instant",
                provider=GroqProvider(api_key=api_key)
            )

            # Initialize the main routing agent with STRICT instructions
            self.router_agent = Agent(model,
            system_prompt=
            """You are a Bible assistant router that ONLY provides general guidance.

            CRITICAL RULES:
            1. NEVER generate or quote Bible verses from your training data
            2. For general help, only provide guidance on HOW to search
            3. Suggest search strategies, not actual Bible content
            4. If asked for specific verses, direct users to search by reference

            You help users understand how to use the Bible search system, but you do NOT provide Bible content yourself."""
        )
    
    async def classify_query(self, user_input: str) -> Dict[str, Any]:
        """
        Classify the type of user query to determine routing
        """
        user_input = user_input.strip()
        
        # Check if it's an explicit Bible reference
        if self.chunker.is_reference_format(user_input):
            return {
                "task_type": "EXACT_LOOKUP",
                "confidence": 0.95,
                "reasoning": "Input matches Bible reference pattern"
            }
        
        # Check for quote-like patterns (text in quotes or verse-like content)
        if (user_input.startswith('"') and user_input.endswith('"')) or \
           any(word in user_input.lower() for word in ['god', 'lord', 'jesus', 'christ', 'love', 'faith', 'hope']):
            return {
                "task_type": "SEMANTIC_SEARCH", 
                "confidence": 0.8,
                "reasoning": "Input appears to be a verse quote or biblical topic"
            }
        
        # Default to semantic search for other biblical queries
        return {
            "task_type": "SEMANTIC_SEARCH",
            "confidence": 0.6,
            "reasoning": "General query - will attempt semantic search"
        }
    
    async def route_query(self, session: AsyncSession, user_input: str, version: str = "kjv") -> Dict[str, Any]:
        """
        Main routing method with strict database validation
        
        Args:
            session: Database session
            user_input: User query text
            version: Bible version to use (e.g., "kjv", "niv", "nkjv", "nlt")
            
        Returns:
            Dict with query results
        """
        try:
            # Validate the version
            from db.models import VERSION_MODELS
            # Allow callers to pass 'auto' for automatic version detection
            if version.lower() != 'auto' and version.lower() not in VERSION_MODELS:
                raise ValueError(f"Unknown Bible version: {version}")
                
            # Normalize version to lowercase
            version = version.lower()
            
            # Classify the query
            classification = await self.classify_query(user_input)
            
            # Route to appropriate handler with version
            if classification["task_type"] == "EXACT_LOOKUP":
                result = await self._handle_exact_lookup(session, user_input, version)
            elif classification["task_type"] == "SEMANTIC_SEARCH":
                result = await self._handle_semantic_search(session, user_input, version)
            else:
                result = await self._handle_general_help(user_input)
                result["version"] = version
            
            # Add classification info to result
            result["classification"] = classification

            # Deduplicate results to avoid duplicate verses from fallback or merges
            if result.get("results"):
                try:
                    refs_before = [getattr(v, 'reference', None) if not isinstance(v, dict) else v.get('reference') for v in result.get('results')]
                    logger.debug("route_query - %d results before dedupe: %s", len(refs_before), refs_before)
                    result["results"] = self._dedupe_verses(result.get("results"))
                    refs_after = [getattr(v, 'reference', None) if not isinstance(v, dict) else v.get('reference') for v in result.get('results')]
                    logger.debug("route_query - %d results after dedupe: %s", len(refs_after), refs_after)
                except Exception:
                    pass
            
            # CRITICAL: Provide deterministic structured AI-style response only
            try:
                result["ai_response_structured"] = self.retriever_agent.format_ai_structured(result)
            except Exception:
                result["ai_response_structured"] = {"type": "error", "query": user_input, "verses": [], "meta": {"found": False}}
            
            return result
            
        except Exception as e:
            return {
                "query_type": "error",
                "query": user_input,
                "results": [],
                "found": False,
                "message": f"Error processing query: {str(e)}",
                "ai_response": "I apologize, but I encountered an error while processing your request. Please try again."
            }
    
    async def _handle_exact_lookup(self, session: AsyncSession, user_input: str, version: str = "kjv") -> Dict[str, Any]:
        """Handle exact Bible reference lookups with validation"""
        result = await self.retriever_agent.retrieve_verses(session, user_input, version)
        
        # Additional validation for exact lookups
        if not result["found"]:
            result["message"] = f"Verse '{user_input}' not found in database ({version.upper()})."
            
        return result
    
    async def _handle_semantic_search(self, session: AsyncSession, user_input: str, version: str = "kjv") -> Dict[str, Any]:
        """Handle semantic search with similarity validation"""
        # If caller explicitly requests auto-detection or the given version yields no results,
        # attempt semantic search across all available versions and pick the best-scoring set.
        from db.models import VERSION_MODELS

        # Normalize version
        version = (version or "").lower()

        # Helper to call retriever directly for semantic search (returns dict with results)
        async def run_for_version(v: str) -> Dict[str, Any]:
            try:
                logger.debug("run_for_version - calling retriever.semantic_search for version=%s", v)
                verses = await self.retriever_agent.semantic_search(session, user_input, v, limit=3)
                if not verses:
                    return {"version": v, "found": False, "results": [], "message": ""}
                # Compute a top similarity score for this version (verses may be dicts with similarity_score)
                top_score = None
                for r in verses:
                    if isinstance(r, dict) and r.get("similarity_score") is not None:
                        try:
                            s = float(r.get("similarity_score"))
                        except Exception:
                            s = None
                    else:
                        s = None
                    if s is not None:
                        top_score = s if top_score is None else max(top_score, s)
                logger.debug("run_for_version - version=%s top_score=%s result_count=%d", v, top_score, len(verses))
                return {"version": v, "found": True, "results": verses, "top_score": top_score or 0.0}
            except Exception as e:
                return {"version": v, "found": False, "results": [], "message": str(e), "top_score": 0.0}

        # If the caller asked for a specific version (not 'auto'), try it first
        tried_versions = []
        if version and version != 'auto':
            tried_versions.append(version)
            primary = await run_for_version(version)
            if primary.get('found'):
                # Attach the detected version and return
                primary['message'] = primary.get('message') or f"Found {len(primary.get('results', []))} relevant verses for: {user_input} ({version.upper()})"
                primary['query'] = user_input
                primary['query_type'] = 'semantic_search'
                return primary

        # Otherwise, or if primary had no results, try auto-detection across all versions
        best = {"version": None, "found": False, "results": [], "top_score": 0.0}
        for v in VERSION_MODELS.keys():
            if v in tried_versions:
                continue
            candidate = await run_for_version(v)
            # prefer higher top_score
            try:
                if candidate.get('found') and float(candidate.get('top_score', 0.0)) > float(best.get('top_score', 0.0)):
                    best = candidate
            except Exception:
                continue

        if not best.get('found'):
            return {
                "query_type": "semantic_search",
                "query": user_input,
                "results": [],
                "found": False,
                "version": version or 'auto',
                "message": f"No sufficiently similar verses found for: {user_input} ({(version or 'AUTO').upper()})",
                "is_range": False
            }

        # Format a successful result mapping to previous return contract
        return {
            "query_type": "semantic_search",
            "query": user_input,
            "results": best.get('results', []),
            "found": True,
            "version": best.get('version'),
            "message": f"Found {len(best.get('results', []))} relevant verses for: {user_input} ({best.get('version').upper()})",
            "is_range": False
        }
    
    async def _handle_general_help(self, user_input: str) -> Dict[str, Any]:
        """Handle general help queries WITHOUT generating Bible content"""
        try:
            # Use the router agent for general questions - NO BIBLE CONTENT
            result = await self.router_agent.run(
                f"The user asked: '{user_input}'. Provide guidance on how to search for Bible verses or use this system. DO NOT quote any Bible verses."
            )

            return {
                "query_type": "general_help",
                "query": user_input,
                "results": [],
                "found": True,
                "message": "General help response provided",
                "ai_response": result.data
            }
        except Exception as e:
            return {
                "query_type": "general_help",
                "query": user_input,
                "results": [],
                "found": False,
                "message": f"Error in general help: {str(e)}",
                "ai_response": "I'm here to help with Bible verse searches. You can search by reference (like 'John 3:16') or by topic/quote. Try being more specific with your search terms."
            }

    def _dedupe_verses(self, verses: List[Any], keep_best_score: bool = True, score_key: str = "similarity_score") -> List[Any]:
        """
        Deduplicate a list of verse objects or dicts.
        - preserves original order
        - if keep_best_score is True and a numeric score is available, keeps the item with highest score
        """
        seen: Dict[tuple, Any] = {}
        order: List[tuple] = []

        def key_of(v):
            try:
                # Use (book, chapter, verse) numeric tuple for stable deduping
                if isinstance(v, dict):
                    return (v.get("book"), v.get("chapter"), v.get("verse"))
                return (getattr(v, "book", None), getattr(v, "chapter", None), getattr(v, "verse", None))
            except Exception:
                return (None, None, None)

        def score_of(v):
            try:
                if isinstance(v, dict):
                    return v.get(score_key) or v.get("score") or v.get("similarity")
                return getattr(v, score_key, None) or getattr(v, "score", None) or getattr(v, "similarity", None)
            except Exception:
                return None

        for v in verses:
            k = key_of(v)
            if k in seen:
                if keep_best_score:
                    try:
                        existing = seen[k]
                        es = score_of(existing)
                        cs = score_of(v)
                        if cs is not None and (es is None or cs > es):
                            seen[k] = v
                    except Exception:
                        pass
                continue
            seen[k] = v
            order.append(k)

        return [seen[k] for k in order]
    
    async def get_suggestions(self, user_input: str) -> List[str]:
        """
        Provide query suggestions based on partial input
        """
        suggestions = []
        user_input_lower = user_input.lower()
        
        # Common Bible references
        common_refs = [
            "John 3:16", "Romans 8:28", "Philippians 4:13", "Psalm 23:1",
            "1 Corinthians 13:4", "Matthew 28:19", "Ephesians 2:8",
            "Isaiah 41:10", "Proverbs 3:5", "Jeremiah 29:11"
        ]
        
        # Common topics - but don't promise specific verses exist
        common_topics = [
            "love", "faith", "hope", "peace", "joy", "forgiveness",
            "salvation", "grace", "prayer", "wisdom", "strength"
        ]
        
        # Add matching references
        for ref in common_refs:
            if user_input_lower in ref.lower():
                suggestions.append(ref)
        
        # Add matching topics
        for topic in common_topics:
            if user_input_lower in topic.lower():
                suggestions.append(f'verses about {topic}')
        
        return suggestions[:5]  # Return top 5 suggestions

    
    async def validate_range_input(self, range_query: str) -> Dict[str, Any]:
        """
        Validate a Bible verse range query
        """
        try:
            # Check if it's a range reference
            if not self.chunker.is_range_reference(range_query):
                return {
                    "valid": False,
                    "message": f"'{range_query}' is not a valid range format. Use format like 'Genesis 1:1-3'",
                    "range_info": None,
                    "error": "Invalid range format"
                }
            
            # Parse the range
            verse_range = self.chunker.parse_range_reference(range_query)
            if not verse_range:
                return {
                    "valid": False,
                    "message": f"Could not parse range '{range_query}'",
                    "range_info": None,
                    "error": "Parse error"
                }
            
            # Get individual references
            individual_refs = verse_range.to_references()
            
            return {
                "valid": True,
                "message": f"Valid range: {range_query}",
                "range_info": {
                    "book": verse_range.book,
                    "chapter": verse_range.chapter,
                    "start_verse": verse_range.start_verse,
                    "end_verse": verse_range.end_verse,
                    "total_verses": len(individual_refs),
                    "individual_references": individual_refs
                },
                "error": None
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error validating range '{range_query}': {str(e)}",
                "range_info": None,
                "error": str(e)
            }
    
    def get_usage_examples(self) -> Dict[str, List[str]]:
        """
        Get examples of different query types
        """
        return {
            "exact_references": [
                "John 3:16",
                "Romans 8:28", 
                "Philippians 4:13",
                "Psalm 23:1",
                "1 Corinthians 13:4",
                "Matthew 28:19",
                "Ephesians 2:8"
            ],
            "verse_ranges": [
                "Genesis 1:1-3",
                "Psalm 23:1-6", 
                "Matthew 5:3-12",
                "1 Corinthians 13:4-8",
                "Romans 8:28-30",
                "John 14:1-6",
                "Ephesians 6:10-18"
            ],
            "semantic_searches": [
                "verses about love",
                "faith and hope", 
                "God's grace",
                "salvation through Christ",
                "peace in troubled times",
                "strength in weakness",
                "forgiveness of sins"
            ],
            "quoted_text": [
                "For God so loved the world",
                "I can do all things through Christ",
                "The Lord is my shepherd",
                "Love is patient love is kind",
                "Be strong and courageous",
                "Cast all your anxiety on him",
                "Trust in the Lord with all your heart"
            ]
        }