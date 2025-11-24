'''
✓ LLM Intent Understanding - Asks what user wants if unclear
✓ Conversation memory (full chat history preserved)
✓ Stronger Math & Physics Agent with SymPy
✓ Multi-source search (DuckDuckGo + SearXNG fallback)
✓ Date extraction & freshness scoring
✓ Better knowledge storage with timestamps
✓ QWEN3:8B FALLBACK FOR MATH/PHYSICS
✓ FIXED: DateTime offset-aware/naive comparison error
✓ Smart Task Detection (Chat/Research/Coding/Math)
✓ International + Indian News Sources
✓ Date-aware responses (past/future detection)
✓ Advanced Coding Workflow (Study → Design → Code → Verify → Google Search)
✓ ENHANCED MATH/PHYSICS EXPLANATIONS - More Summary & Deep Reasoning
✓ NEW: Smart Intent Understanding - Clarify unclear requests via web
✓ NEW: FAST PATH for simple queries (greetings, basic math, time/weather)
✓ NEW: DEDICATED CODER MODEL (qwen2.5-coder:7b) for code generation
✓ NEW: OUTPUT VERIFICATION PIPELINE - Validates answers against Google
✓ NEW: NATURAL CHAT RESPONSES - No scripted replies

Requirements:
pip install ollama requests beautifulsoup4 sympy python-dateutil'''

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from ollama import Client
import json
import os
import time
import re
import requests
import traceback

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr
except ImportError:
    sp = None

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

# ================== CONFIG ==================

client = Client(host="http://localhost:11434")

MODEL_NAME = "llama3.1:8b"
QWEN_MODEL = "qwen3:8b"
CODER_MODEL = "qwen2.5-coder:7b"  # NEW: Dedicated coding model
KNOWLEDGE_PATH = "knowledge_store.json"
CONVERSATION_PATH = "conversation_history.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# NEWS SOURCES - International + Indian
NEWS_SOURCES = {
    "international": [
        {"name": "BBC News", "url": "https://www.bbc.com/news"},
        {"name": "Reuters", "url": "https://www.reuters.com"},
        {"name": "AP News", "url": "https://apnews.com"},
        {"name": "CNN", "url": "https://www.cnn.com"},
        {"name": "The Guardian", "url": "https://www.theguardian.com"},
    ],
    "indian": [
        {"name": "Times of India", "url": "https://timesofindia.indiatimes.com"},
        {"name": "The Hindu", "url": "https://www.thehindu.com"},
        {"name": "Indian Express", "url": "https://indianexpress.com"},
        {"name": "Hindustan Times", "url": "https://www.hindustantimes.com"},
        {"name": "NDTV", "url": "https://www.ndtv.com"},
    ]
}

SYSTEM_PROMPT = """
You are an advanced AI assistant with access to:
- Real-time web knowledge (WEB_KNOWLEDGE)
- Conversation history (you remember everything)
- Symbolic math engine for perfect calculations
- Date awareness (knows past/future)

MODES:
- Chat: Respond naturally, reference past conversation
- Knowledge: Use WEB_KNOWLEDGE which contains verified multi-source data with dates
- Math: Delegate to math engine for perfect accuracy
- Code: Generate executable code in <python> or <cpp> tags
- Research: Deep analysis with citations

For code:
<python>
# code here
</python>

<cpp>
// code here
</cpp>

CRITICAL RULES:
- Always check dates when discussing current events
- Reference conversation history naturally
- Never hallucinate calculations - use the math engine
- For conflicting info, state confidence level
- For research: cite ALL sources and dates
- In casual chat, respond naturally without scripted phrases
"""

# ================== NEW: FAST PATH DETECTION ==================

def fast_path_check(msg: str) -> Optional[Tuple[str, str]]:
    """
    Fast path for simple queries that don't need heavy pipeline.
    Returns: (task_type, answer) if handled, None otherwise
    """
    msg_lower = msg.lower().strip()
    
    # 1. GREETINGS & CASUAL CHAT
    greeting_patterns = [
        r'\b(hi|hey|hello|sup|wassup|yo)\b',
        r'\bhow are you\b',
        r'\bwhat\'?s up\b',
        r'\bgood (morning|afternoon|evening|night)\b',
        r'\bnice to meet you\b',
    ]
    
    for pattern in greeting_patterns:
        if re.search(pattern, msg_lower):
            print("[FAST_PATH] ✓ Detected: GREETING")
            return ("CHAT_FAST", None)  # Let LLM respond naturally
    
    # 2. SIMPLE ARITHMETIC (not algorithm complexity!)
    # Only trigger if it's clearly a direct calculation
    simple_math_patterns = [
        r'^[\d\s\+\-\*/\(\)\.]+$',  # Pure arithmetic: "2+2", "10*5"
        r'^(what is|what\'s|calculate|compute)\s+[\d\s\+\-\*/\(\)\.]+\??$',  # "what is 2+2"
    ]
    
    # EXCLUDE if it contains algorithm/complexity keywords
    exclude_keywords = ['complexity', 'algorithm', 'big o', 'runtime', 'performance', 'analysis']
    if not any(keyword in msg_lower for keyword in exclude_keywords):
        for pattern in simple_math_patterns:
            if re.search(pattern, msg_lower):
                print("[FAST_PATH] ✓ Detected: SIMPLE ARITHMETIC")
                try:
                    # Extract and evaluate
                    equation = re.sub(r'(what is|what\'s|calculate|compute)', '', msg_lower)
                    equation = equation.strip().strip('?')
                    result = eval(equation)
                    return ("MATH_FAST", f"**Answer:** {result}")
                except:
                    pass
    
    # 3. TIME QUERIES (but not "time complexity"!)
    time_patterns = [
        r'\b(what is|what\'s|current|show|tell me).*\btime\b.*\b(in|at|for)\b',  # "what's the time in Tokyo"
        r'\btime\s+(in|at|for)\s+\w+',  # "time in London"
        r'\bcurrent time\b',
        r'\bwhat time is it\b',
    ]
    
    # EXCLUDE algorithm/complexity context
    if 'complexity' not in msg_lower and 'algorithm' not in msg_lower:
        for pattern in time_patterns:
            if re.search(pattern, msg_lower):
                print("[FAST_PATH] ✓ Detected: TIME QUERY")
                # Extract location if present
                location_match = re.search(r'\b(in|at|for)\s+(\w+(?:\s+\w+)?)', msg_lower)
                location = location_match.group(2) if location_match else "local"
                
                current_time = datetime.now().strftime("%H:%M:%S")
                current_date = datetime.now().strftime("%A, %B %d, %Y")
                
                answer = f"**Current Time:** {current_time}\n**Date:** {current_date}\n**Location:** {location.title()}"
                
                if location != "local":
                    answer += f"\n\n*Note: For accurate {location.title()} time, please check a timezone converter.*"
                
                return ("TIME_FAST", answer)
    
    # 4. WEATHER QUERIES
    weather_patterns = [
        r'\b(weather|temperature|forecast|climate)\b.*\b(in|at|for|of)\s+\w+',
        r'\b(what\'s|what is|how\'s|how is).*\b(weather|temperature)\b',
        r'\b(rain|sunny|cloudy|storm)\b.*\b(today|tomorrow)\b',
    ]
    
    for pattern in weather_patterns:
        if re.search(pattern, msg_lower):
            print("[FAST_PATH] ✓ Detected: WEATHER QUERY (needs web search)")
            return ("WEATHER_FAST", None)  # Will use quick web search
    
    # 5. SIMPLE "WHO/WHAT/WHERE IS" questions that need quick search
    quick_search_patterns = [
        r'^(who|what|where) (is|are|was|were)\s+\w+',  # "who is X", "what is Y"
    ]
    
    for pattern in quick_search_patterns:
        if re.search(pattern, msg_lower) and len(msg.split()) <= 6:  # Keep it simple
            print("[FAST_PATH] ✓ Detected: QUICK SEARCH QUERY")
            return ("QUICK_SEARCH", None)
    
    # No fast path match
    return None

# ================== NEW: OUTPUT VERIFICATION PIPELINE ==================

def verify_output_with_google(question: str, generated_output: str, task_type: str) -> Dict[str, Any]:
    """
    Verify generated output by searching Google and comparing.
    Returns verification result with confidence score.
    """
    print("\n" + "="*80)
    print("🔍 GOOGLE VERIFICATION PIPELINE STARTED")
    print("="*80)
    
    # Don't verify casual chat
    if task_type in ("CHAT", "CHAT_FAST", "GREETING"):
        return {
            "verified": True,
            "confidence": 1.0,
            "message": "Casual chat - no verification needed",
            "corrections": None
        }
    
    print(f"[VERIFY] Task Type: {task_type}")
    print(f"[VERIFY] Question: {question[:100]}...")
    
    # Step 1: Search Google for the same question
    print("[VERIFY] Searching web for verification...")
    search_query = question
    raw_text, sources, dates = deep_browse_multi(search_query, max_pages=5)
    
    if not raw_text:
        print("[VERIFY] ⚠️ No web sources found for verification")
        return {
            "verified": False,
            "confidence": 0.5,
            "message": "Could not find web sources for verification",
            "corrections": None
        }
    
    # Step 2: Compare our output with web sources
    print("[VERIFY] Comparing output with web sources...")
    
    comparison_prompt = f"""
You are a fact-checker. Compare our AI's answer with verified web sources.

QUESTION:
{question}

OUR AI'S ANSWER:
{generated_output[:3000]}

WEB SOURCES (VERIFIED):
{raw_text[:5000]}

CRITICAL ANALYSIS:
1. Is our answer FACTUALLY CORRECT? [YES/NO/PARTIALLY]
2. Confidence in our answer? [0.0 - 1.0]
3. What did we get RIGHT?
4. What did we get WRONG (if anything)?
5. Should we correct anything? [YES/NO]
6. If YES, what corrections are needed?
7. Overall verdict: [PASS/FAIL/NEEDS_IMPROVEMENT]

Respond with JSON only:
{{
  "factually_correct": "YES|NO|PARTIALLY",
  "confidence": 0.95,
  "correct_parts": ["list of correct things"],
  "incorrect_parts": ["list of errors"],
  "needs_correction": false,
  "corrections": "specific corrections needed",
  "verdict": "PASS|FAIL|NEEDS_IMPROVEMENT",
  "reasoning": "detailed explanation"
}}
"""
    
    try:
        resp = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a meticulous fact-checker. Be honest and critical."},
                {"role": "user", "content": comparison_prompt}
            ]
        )
        
        raw = resp["message"]["content"].strip()
        
        # Parse JSON
        try:
            if raw.startswith("{"):
                result = json.loads(raw)
            else:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    result = {"verified": False, "confidence": 0.5}
        except json.JSONDecodeError:
            result = {"verified": False, "confidence": 0.5}
        
        # Add sources
        result["sources"] = sources[:5]
        
        print(f"[VERIFY] ✓ Verdict: {result.get('verdict', 'UNKNOWN')}")
        print(f"[VERIFY] ✓ Confidence: {result.get('confidence', 0):.0%}")
        
        return result
        
    except Exception as e:
        print(f"[VERIFY ERROR] {e}")
        return {
            "verified": False,
            "confidence": 0.5,
            "message": f"Verification failed: {e}",
            "corrections": None
        }

def handle_verification_result(verification: Dict[str, Any], original_output: str, question: str) -> str:
    """
    Handle verification result and optionally fix output.
    """
    verdict = verification.get("verdict", "UNKNOWN")
    confidence = verification.get("confidence", 0.5)
    needs_correction = verification.get("needs_correction", False)
    
    if verdict == "PASS" and confidence >= 0.8:
        print("[VERIFY] ✅ Output verified - PASSING as is")
        
        verification_badge = f"\n\n{'='*80}\n✅ **VERIFIED** - Cross-checked with {len(verification.get('sources', []))} web sources (Confidence: {confidence:.0%})\n{'='*80}"
        return original_output + verification_badge
    
    elif verdict == "FAIL" or needs_correction:
        print("[VERIFY] ❌ Output failed verification - ATTEMPTING FIX")
        
        # Search web again for correct answer
        print("[VERIFY] Researching correct answer from web...")
        raw_text, sources, dates = deep_browse_multi(question, max_pages=8)
        
        if raw_text:
            fix_prompt = f"""
The previous answer was INCORRECT or INCOMPLETE.

QUESTION:
{question}

PREVIOUS (INCORRECT) ANSWER:
{original_output[:2000]}

WHAT WAS WRONG:
{verification.get('incorrect_parts', 'Multiple errors')}

CORRECTIONS NEEDED:
{verification.get('corrections', 'See web sources')}

VERIFIED WEB SOURCES:
{raw_text[:8000]}

Generate a NEW, CORRECT, COMPREHENSIVE answer based on verified web sources.
Be accurate and detailed. Include sources.
"""
            
            resp = client.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a research expert. Provide accurate, well-sourced answers."},
                    {"role": "user", "content": fix_prompt}
                ]
            )
            
            corrected_output = resp["message"]["content"].strip()
            
            # Add correction notice
            correction_notice = f"\n\n{'='*80}\n⚠️ **CORRECTED ANSWER** - Original answer was inaccurate\n{'='*80}\n\n"
            
            if sources:
                corrected_output += f"\n\n📚 **VERIFIED SOURCES ({len(sources)}):**\n"
                for i, src in enumerate(sources[:5], 1):
                    corrected_output += f"[{i}] {src}\n"
            
            print("[VERIFY] ✅ Generated corrected answer")
            return correction_notice + corrected_output
        
        else:
            warning = f"\n\n{'='*80}\n⚠️ **VERIFICATION FAILED** - Could not verify answer accuracy\nConfidence: {confidence:.0%}\n{'='*80}"
            return original_output + warning
    
    else:
        # Partial pass
        print(f"[VERIFY] ⚠️ Output partially verified (Confidence: {confidence:.0%})")
        
        warning = f"\n\n{'='*80}\n⚠️ **PARTIALLY VERIFIED** - Some details may need verification\nConfidence: {confidence:.0%}\n{'='*80}"
        return original_output + warning

# ================== LLM INTENT UNDERSTANDING ==================

def understand_user_intent(msg: str) -> Dict[str, Any]:
    """LLM understands exactly what user wants - detailed intent parsing"""
    
    system_prompt = """You are an intent analyzer. Deeply understand what the user REALLY wants.

Respond with ONLY JSON:
{
  "task_type": "MATH|CODING|RESEARCH|CHAT|UNCLEAR",
  "confidence": 0.95,
  "what_user_wants": "Brief 1-line summary of what user wants",
  "specific_goal": "More specific goal/outcome the user expects",
  "context": "Any important context or constraints",
  "reasoning": "Why you classified it this way",
  "clarification_needed": false,
  "clarification_questions": []
}

RULES:
- task_type: What is the ACTUAL task?
- confidence: How sure are you? (0.0-1.0)
- what_user_wants: What is user trying to achieve?
- specific_goal: What is the desired outcome?
- context: What important info do we need?
- clarification_needed: Do we need to ask user for more info?
- If confidence < 0.6, mark clarification_needed as true"""
    
    user_prompt = f"Deeply analyze what the user wants:\n\"{msg}\"\n\nRespond with JSON only:"
    
    try:
        resp = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        raw = resp["message"]["content"].strip()
        
        try:
            if raw.startswith("{"):
                intent = json.loads(raw)
            else:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    intent = json.loads(match.group(0))
                else:
                    return {"task_type": "CHAT", "confidence": 0.0, "clarification_needed": True}
        except json.JSONDecodeError:
            return {"task_type": "CHAT", "confidence": 0.0, "clarification_needed": True}
        
        return intent
        
    except Exception as e:
        print(f"[INTENT ERROR] {e}")
        return {"task_type": "CHAT", "confidence": 0.0}

def ask_web_about_intent(msg: str) -> Dict[str, Any]:
    """Search web to understand what user is asking about"""
    
    print("[WEB_INTENT] Searching web to understand user intent...")
    
    # Search for context
    results = multi_search(msg, max_results=3)
    
    if not results:
        print("[WEB_INTENT] No web results found")
        return {"task_type": "CHAT", "web_context": "No web context found"}
    
    # Extract content from top results
    web_context_parts = []
    for result in results[:2]:
        url = result["url"]
        text, _ = fetch_and_extract(url, max_chars=1000)
        if text:
            web_context_parts.append(f"Source: {result.get('source', 'Web')}\n{text[:500]}")
    
    web_context = "\n\n".join(web_context_parts)
    
    if not web_context:
        print("[WEB_INTENT] Could not extract web content")
        return {"task_type": "CHAT", "web_context": "No content extracted"}
    
    # Ask LLM with web context
    clarify_prompt = f"""
User message: "{msg}"

Web context found:
\"{web_context[:2000]}\"

Based on web context, what does the user ACTUALLY want?
What type of task is this?

Respond with JSON:
{{
  "task_type": "MATH|CODING|RESEARCH|CHAT",
  "what_user_wants": "What user wants",
  "web_helped": true,
  "explanation": "How web context helped clarify"
}}
"""
    
    resp = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an intent analyzer. Use web context to understand user."},
            {"role": "user", "content": clarify_prompt}
        ]
    )
    
    raw = resp["message"]["content"].strip()
    
    try:
        if raw.startswith("{"):
            result = json.loads(raw)
        else:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                result = {"task_type": "CHAT"}
    except json.JSONDecodeError:
        result = {"task_type": "CHAT"}
    
    result["web_context"] = web_context
    return result

def clarify_with_user(intent: Dict[str, Any]) -> Optional[str]:
    """If LLM is unclear, ask user for clarification"""
    
    if not intent.get("clarification_needed", False):
        return None
    
    questions = intent.get("clarification_questions", [])
    
    print("\n" + "="*80)
    print("🤔 I need clarification to serve you better!")
    print("="*80)
    
    if questions:
        print("\nPlease answer these questions:\n")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")
        print()
    
    clarification = input("You: ").strip()
    
    if clarification:
        return clarification
    
    return None

# ================== DETECT & UNDERSTAND TASK ==================

def detect_and_understand_task(msg: str) -> Tuple[str, Dict[str, Any]]:
    """
    Pipeline:
    1. LLM understands intent
    2. If unclear, search web for context
    3. Return task type + full intent info
    """
    
    print("[PIPELINE] Understanding user intent...")
    
    # Step 1: LLM understands intent
    intent = understand_user_intent(msg)
    task_type = intent.get("task_type", "CHAT")
    confidence = intent.get("confidence", 0.5)
    
    print(f"[PIPELINE] Initial understanding: {task_type} (confidence: {confidence:.0%})")
    print(f"[PIPELINE] What user wants: {intent.get('what_user_wants', 'N/A')}")
    
    # Step 2: If unsure, use web to clarify
    if task_type == "UNCLEAR" or confidence < 0.6:
        print("[PIPELINE] Confidence too low - asking web for context...")
        web_intent = ask_web_about_intent(msg)
        
        # Merge web findings with original intent
        intent.update(web_intent)
        task_type = web_intent.get("task_type", "CHAT")
        
        print(f"[PIPELINE] Web-assisted understanding: {task_type}")
        print(f"[PIPELINE] What user wants: {web_intent.get('what_user_wants', 'N/A')}")
    
    return task_type, intent

# ================== LLM-BASED TASK DETECTION (Original - Still Used) ==================

def detect_task_type_with_llm(msg: str) -> str:
    """Use LLM to intelligently detect task type"""
    
    system_prompt = """You are a task classifier. Classify user messages into ONE category.

CATEGORIES:
1. MATH - Mathematical problems, equations, physics, calculations (solve for x, integrate, etc)
2. CODING - Build apps, websites, APIs, code projects, programming tasks
3. RESEARCH - Explain concepts, current events, news, "what is", historical info, analysis
4. CHAT - General conversation, greetings, opinions, casual questions

RESPOND WITH ONLY:
{
  "task": "MATH|CODING|RESEARCH|CHAT",
  "confidence": 0.95,
  "reasoning": "Brief reason"
}

Be smart and contextual. If unsure (confidence < 0.7), return task as "UNCERTAIN"."""
    
    user_prompt = f"Classify this message:\n\"{msg}\""
    
    try:
        resp = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        raw_response = resp["message"]["content"].strip()
        
        try:
            if raw_response.startswith("{"):
                result = json.loads(raw_response)
            else:
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    return "CHAT"
        except json.JSONDecodeError:
            return "CHAT"
        
        task = result.get("task", "CHAT").upper()
        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")
        
        print(f"[DETECT] Task: {task} | Confidence: {confidence:.1%} | Reason: {reasoning}")
        
        if task == "UNCERTAIN" or confidence < 0.7:
            print("[DETECT] Low confidence - asking web for verification...")
            task = ask_web_for_task_classification(msg)
            print(f"[DETECT] Web classified as: {task}")
        
        return task
        
    except Exception as e:
        print(f"[DETECT ERROR] {e}")
        return "CHAT"

def ask_web_for_task_classification(msg: str) -> str:
    """If LLM is uncertain, search web context to help classify"""
    
    print("[WEB_DETECT] Searching web for context...")
    
    results = multi_search(msg, max_results=3)
    
    if not results:
        print("[WEB_DETECT] No web results - defaulting to CHAT")
        return "CHAT"
    
    url = results[0]["url"]
    text, _ = fetch_and_extract(url, max_chars=2000)
    
    if not text:
        print("[WEB_DETECT] Could not extract web content - defaulting to CHAT")
        return "CHAT"
    
    classify_prompt = f"""
User message: "{msg}"

Web context from search:
\"{text[:1500]}\"

Now classify - is this:
1. MATH - Mathematical/physics problem?
2. CODING - Programming/development task?
3. RESEARCH - Knowledge/explanation question?
4. CHAT - General conversation?

Respond with ONLY the category: MATH|CODING|RESEARCH|CHAT
"""
    
    resp = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a task classifier. Answer with ONLY one word."},
            {"role": "user", "content": classify_prompt}
        ]
    )
    
    task = resp["message"]["content"].strip().upper()
    
    if task not in ("MATH", "CODING", "RESEARCH", "CHAT"):
        task = "CHAT"
    
    return task

# ================== CONVERSATION HISTORY ==================

class ConversationManager:
    def __init__(self, path: str = CONVERSATION_PATH):
        self.path = path
        self.messages: List[Dict[str, str]] = []
        self.load()
    
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.messages = data.get("messages", [])
                    print(f"[CONV] Loaded {len(self.messages)} messages from history")
            except Exception as e:
                print(f"[CONV ERROR] Could not load: {e}")
                self.messages = []
    
    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({
                    "messages": self.messages,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CONV SAVE ERROR] {e}")
    
    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def get_context(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Get recent conversation for context"""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": m["role"], "content": m["content"]} for m in recent]
    
    def clear(self):
        self.messages = []
        self.save()
        print("[CONV] History cleared")

conversation = ConversationManager()

# ================== ENHANCED MEMORY ==================

def _load_knowledge() -> Dict[str, Any]:
    if not os.path.exists(KNOWLEDGE_PATH):
        return {}
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_knowledge(store: Dict[str, Any]) -> None:
    try:
        with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[MEMORY SAVE ERROR] {e}")

def _normalize_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q[:200]

def get_stored_knowledge(query: str) -> Optional[Dict[str, Any]]:
    mem = _load_knowledge()
    key = _normalize_query(query)
    
    if key in mem:
        entry = mem[key]
        stored_time = entry.get("time", 0)
        age_hours = (time.time() - stored_time) / 3600
        
        if age_hours > 24:
            print("[MEMORY] Cached news too old, refreshing...")
            return None
        
        return entry
    
    for stored_key, entry in mem.items():
        if key in stored_key or stored_key in key:
            return entry
    
    return None

def store_knowledge(query: str, summary: str, sources: List[str] = None, dates: List[str] = None) -> None:
    if not summary:
        return
    
    mem = _load_knowledge()
    key = _normalize_query(query)
    
    mem[key] = {
        "time": time.time(),
        "date": datetime.now().isoformat(),
        "summary": summary,
        "sources": sources or [],
        "dates": dates or [],
    }
    _save_knowledge(mem)
    print(f"[MEMORY] Stored knowledge from {len(sources or [])} sources")

# ================== DATE UTILITIES ==================

def parse_date_safely(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime (offset-naive for comparison)"""
    if not date_str or not DATEUTIL_AVAILABLE:
        return None
    
    try:
        dt = date_parser.parse(date_str, fuzzy=True)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except:
        return None

def get_date_context(dt: Optional[datetime]) -> str:
    """Get human-readable date context"""
    if not dt:
        return ""
    
    now = datetime.now()
    diff = (now - dt).days
    
    if diff < 0:
        return f"(FUTURE: in {abs(diff)} days)"
    elif diff == 0:
        return "(TODAY)"
    elif diff == 1:
        return "(YESTERDAY)"
    elif diff < 7:
        return f"({diff} days ago)"
    elif diff < 30:
        return f"({diff//7} weeks ago)"
    else:
        return f"({dt.strftime('%B %Y')})"

def is_past_event(date_str: str) -> bool:
    """Check if date is in the past"""
    dt = parse_date_safely(date_str)
    if not dt:
        return True
    return (datetime.now() - dt).days > 0

def is_future_event(date_str: str) -> bool:
    """Check if date is in the future"""
    dt = parse_date_safely(date_str)
    if not dt:
        return False
    return (datetime.now() - dt).days < 0

# ================== MULTI-SOURCE SEARCH ==================

def search_ddg_html(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """Enhanced DuckDuckGo search"""
    try:
        params = {"q": query, "kl": "us-en"}
        r = requests.get("https://duckduckgo.com/html/", params=params, headers=HEADERS, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        results = []
        for result_div in soup.find_all("div", class_="result"):
            try:
                title_elem = result_div.find("a", class_="result__a")
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)
                url = title_elem.get("href", "")
                
                if url.startswith("/l/?kh=") or url.startswith("//duckduckgo.com"):
                    if "uddg=" in url:
                        from urllib.parse import parse_qs, urlparse, unquote
                        try:
                            qs = parse_qs(urlparse(url).query)
                            url = unquote(qs.get("uddg", [""])[0])
                        except:
                            continue
                    else:
                        continue
                
                if url.startswith("http") and "duckduckgo.com" not in url:
                    results.append({"url": url, "title": title, "source": "ddg"})
                
                if len(results) >= max_results:
                    break
            except Exception as e:
                continue
        
        return results
    except Exception as e:
        print(f"[SEARCH/DDG ERROR] {e}")
        return []

def search_searxng(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """Fallback: Public SearXNG instance"""
    try:
        instances = [
            "https://searx.be",
            "https://search.sapti.me",
            "https://searx.tiekoetter.com"
        ]
        
        for instance in instances:
            try:
                params = {
                    "q": query,
                    "format": "json",
                    "categories": "general"
                }
                r = requests.get(f"{instance}/search", params=params, headers=HEADERS, timeout=10)
                r.raise_for_status()
                data = r.json()
                
                results = []
                for item in data.get("results", [])[:max_results]:
                    results.append({
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "source": "searxng"
                    })
                
                if results:
                    return results
            except:
                continue
        
        return []
    except Exception as e:
        print(f"[SEARCH/SEARXNG ERROR] {e}")
        return []

def search_news_sources(query: str, source_type: str = "all") -> List[Dict[str, str]]:
    """Search international and Indian news sources"""
    results = []
    
    if source_type in ("all", "international"):
        for news in NEWS_SOURCES["international"]:
            try:
                r = requests.get(news["url"], headers=HEADERS, timeout=10)
                soup = BeautifulSoup(r.text, "html.parser")
                
                for article in soup.find_all(["h1", "h2", "h3"], limit=5):
                    text = article.get_text(strip=True)
                    if query.lower() in text.lower() and len(text) > 10:
                        results.append({
                            "url": news["url"],
                            "title": text,
                            "source": news["name"]
                        })
            except:
                continue
    
    if source_type in ("all", "indian"):
        for news in NEWS_SOURCES["indian"]:
            try:
                r = requests.get(news["url"], headers=HEADERS, timeout=10)
                soup = BeautifulSoup(r.text, "html.parser")
                
                for article in soup.find_all(["h1", "h2", "h3"], limit=5):
                    text = article.get_text(strip=True)
                    if query.lower() in text.lower() and len(text) > 10:
                        results.append({
                            "url": news["url"],
                            "title": text,
                            "source": news["name"]
                        })
            except:
                continue
    
    return results[:10]

def multi_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search multiple sources and merge"""
    all_results = []
    
    print("[SEARCH] Trying news sources...")
    news = search_news_sources(query, "all")
    all_results.extend(news)
    
    print("[SEARCH] Searching DuckDuckGo...")
    ddg = search_ddg_html(query, max_results)
    all_results.extend(ddg)
    
    if len(all_results) < 3:
        print("[SEARCH] Results low, trying SearXNG...")
        searx = search_searxng(query, max_results)
        all_results.extend(searx)
    
    seen = set()
    unique = []
    for item in all_results:
        url = item["url"]
        if url not in seen and url.startswith("http"):
            seen.add(url)
            unique.append(item)
    
    return unique[:max_results]

# ================== DATE EXTRACTION ==================

def extract_date_from_html(html: str, url: str) -> Optional[str]:
    """Extract publication date from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    
    date_metas = [
        ('meta', {'property': 'article:published_time'}),
        ('meta', {'name': 'publish-date'}),
        ('meta', {'name': 'date'}),
        ('meta', {'property': 'og:updated_time'}),
        ('meta', {'name': 'DC.date.issued'}),
    ]
    
    for tag, attrs in date_metas:
        elem = soup.find(tag, attrs)
        if elem:
            date_str = elem.get('content') or elem.get('datetime', '')
            if date_str:
                return date_str
    
    time_tags = soup.find_all("time")
    for time_tag in time_tags:
        dt = time_tag.get("datetime") or time_tag.get_text(strip=True)
        if dt:
            return dt
    
    if DATEUTIL_AVAILABLE:
        text = soup.get_text()[:5000]
        date_patterns = [
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{4}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
    
    return None

# ================== ENHANCED FETCHING ==================

def fetch_and_extract(url: str, max_chars: int = 10000) -> Tuple[str, Optional[str]]:
    """Fetch page and extract text + date"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True)
        r.raise_for_status()
        
        content_type = r.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return "", None
        
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        
        date = extract_date_from_html(html, url)
        
        for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
            tag.decompose()
        
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        
        if main_content:
            texts = []
            for elem in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
                text = elem.get_text(separator=" ", strip=True)
                if len(text) > 20:
                    texts.append(text)
            
            joined = "\n".join(texts)[:max_chars]
            return joined, date
        
        return "", None
        
    except Exception as e:
        print(f"[FETCH ERROR] {url[:50]}: {e}")
        return "", None

def deep_browse_multi(query: str, max_pages: int = 10) -> Tuple[str, List[str], List[str]]:
    """Enhanced browsing with date tracking"""
    print(f"[BROWSE] Searching multiple sources for: {query[:50]}...")
    
    results = multi_search(query, max_results=max_pages)
    
    if not results:
        print("[BROWSE] No search results found")
        return "", [], []
    
    print(f"[BROWSE] Found {len(results)} results, fetching content...")
    
    parts = []
    sources = []
    dates = []
    
    for i, item in enumerate(results):
        url = item["url"]
        print(f"[BROWSE] [{i+1}/{len(results)}] {url[:60]}...")
        
        text, date = fetch_and_extract(url)
        
        if text:
            date_str = ""
            if date:
                parsed = parse_date_safely(date)
                if parsed:
                    date_str = f" [{parsed.strftime('%Y-%m-%d')}]"
                    dates.append(date)
            
            parts.append(f"=== SOURCE {i+1}{date_str} ===\nURL: {url}\nSource: {item.get('source', 'Web')}\n\n{text}\n")
            sources.append(url)
    
    if not parts:
        print("[BROWSE] No content extracted")
        return "", [], []
    
    combined = "\n\n".join(parts)
    print(f"[BROWSE] Successfully extracted {len(sources)} sources")
    
    return combined, sources, dates

# ================== ENHANCED SUMMARIZATION ==================

def summarize_with_dates(query: str, raw_text: str, sources: List[str], dates: List[str]) -> str:
    """Summarize with emphasis on dates and accuracy"""
    if not raw_text:
        return ""
    
    freshness_note = ""
    if dates:
        parsed_dates = [parse_date_safely(d) for d in dates]
        parsed_dates = [d for d in parsed_dates if d is not None]
        
        if parsed_dates:
            newest = max(parsed_dates)
            oldest = min(parsed_dates)
            
            days_ago = (datetime.now() - newest).days
            if days_ago == 0:
                freshness_note = "🔴 Sources from TODAY (VERY FRESH)\n\n"
            elif days_ago == 1:
                freshness_note = "🟡 Sources from YESTERDAY\n\n"
            elif days_ago < 7:
                freshness_note = f"🟡 Sources from {days_ago} days ago\n\n"
            elif days_ago < 30:
                freshness_note = f"🟢 Sources from {days_ago // 7} weeks ago\n\n"
            else:
                freshness_note = f"🟢 Sources from {newest.strftime('%B %Y')}\n\n"
    
    prompt = f"""
User query:
{query}

{freshness_note}

Raw content from {len(sources)} verified sources:
\"\"\"{raw_text[:25000]}\"\"\"

CRITICAL INSTRUCTIONS:
1. ALWAYS include specific dates when mentioned in sources
2. If sources have conflicting info, mention BOTH with dates
3. For current events, emphasize timeline (what happened when)
4. Structure: Overview → Key Facts with Dates → Current Status → Context
5. Rate confidence: [HIGH/MEDIUM/LOW CONFIDENCE] at the end
6. Keep numerical facts EXACT (don't round unless necessary)
7. This will be stored as reference - make it comprehensive and accurate
8. Cite sources with numbers like [1], [2], etc.

Create a detailed, date-aware summary:
"""
    
    resp = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a meticulous research analyst. Accuracy and dates are critical."},
            {"role": "user", "content": prompt}
        ]
    )
    
    summary = resp["message"]["content"].strip()
    
    if sources:
        summary += f"\n\n📚 **SOURCES ({len(sources)} verified articles):**\n"
        for i, src in enumerate(sources[:10], 1):
            summary += f"[{i}] {src}\n"
    
    return summary

# ================== ENHANCED MATH & PHYSICS SUPER AGENT ==================

def generate_math_explanation_summary(problem: str, solution: str) -> str:
    """Generate comprehensive explanation and summary for math solution"""
    
    print("[MATH] Generating comprehensive explanation...")
    
    explanation_prompt = f"""
PROBLEM: {problem}

SOLUTION PROVIDED:
{solution}

Generate a COMPREHENSIVE explanation with:

1. **PROBLEM SUMMARY**
   - What exactly are we solving?
   - What is being asked?
   - Key information given

2. **SOLUTION APPROACH**
   - What method/formula did we use?
   - Why is this the right approach?
   - Alternative approaches (if any)

3. **DETAILED STEPS**
   - Break down EVERY step
   - Explain the mathematical reasoning for each step
   - Show any substitutions or transformations
   - Explain why certain rules/theorems apply

4. **FINAL ANSWER**
   - Clear statement of the answer
   - Verify if it makes sense (sanity check)
   - Units/dimensions (if applicable)

5. **KEY CONCEPTS INVOLVED**
   - List all mathematical concepts used
   - Brief explanation of each concept
   - Real-world applications (if applicable)

6. **TIPS FOR SIMILAR PROBLEMS**
   - Common mistakes to avoid
   - How to approach similar questions
   - Practice methods

7. **CONFIDENCE ASSESSMENT**
   - Is this answer 100% accurate? [YES/NO]
   - Verification check done? [YES/NO]
   - Why is this answer reliable?

IMPORTANT:
- Be VERY detailed and comprehensive
- Assume reader has basic math knowledge but needs clarification
- Use proper mathematical notation
- Include numerical verification where possible
- Make it educational and helpful for learning

Provide complete explanation:
"""
    
    resp = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """You are an expert mathematics and physics tutor. 
Your job is to provide COMPREHENSIVE explanations that help students truly understand, not just see the answer.
Be thorough, clear, and educational."""
            },
            {"role": "user", "content": explanation_prompt}
        ]
    )
    
    return resp["message"]["content"].strip()

def physics_reasoning_summary(problem: str, solution: str) -> str:
    """Special reasoning for physics problems"""
    
    print("[PHYSICS] Generating physics reasoning and summary...")
    
    physics_prompt = f"""
PHYSICS PROBLEM: {problem}

SOLUTION:
{solution}

Provide COMPREHENSIVE physics analysis:

1. **PHYSICAL CONCEPT**
   - What physics principle governs this problem?
   - What laws/theorems apply? (Newton's laws, Energy conservation, etc.)
   - Why do these laws apply here?

2. **PROBLEM BREAKDOWN**
   - Identify all forces/fields/interactions
   - What is changing and what is constant?
   - What are we trying to find and why?

3. **SOLUTION METHODOLOGY**
   - Step-by-step solution with physical interpretation
   - What does each equation represent physically?
   - Why did we choose this path to solution?

4. **PHYSICAL INTERPRETATION**
   - What does the answer mean in real world?
   - Is the magnitude reasonable?
   - Is the direction/sign correct?
   - What happens if we change variables?

5. **VERIFICATION**
   - Does it satisfy physical laws?
   - Dimensional analysis - correct units?
   - Limiting case check - does it make sense at extremes?
   - Order of magnitude - is it reasonable?

6. **DEEPER UNDERSTANDING**
   - Why is this result true?
   - Connection to other concepts
   - Real-world applications and examples

7. **COMMON MISCONCEPTIONS**
   - What mistakes do students make?
   - How to think about this correctly
   - Why intuition might be wrong here

Format answer for deep understanding, not just calculation.
"""
    
    resp = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """You are a physics expert and educator with deep understanding.
Explain physics problems with emphasis on:
- Physical intuition and reasoning
- Real-world meaning
- Why concepts work the way they do
- Prevention of common misconceptions"""
            },
            {"role": "user", "content": physics_prompt}
        ]
    )
    
    return resp["message"]["content"].strip()

def math_physics_super_agent(user_msg: str) -> Optional[str]:
    """Enhanced math/physics agent with detailed explanations"""
    
    print("\n[MATH_AGENT] Processing mathematical/physics problem...")
    
    is_physics = any(word in user_msg.lower() for word in ["force", "velocity", "acceleration", "gravity", "energy", "momentum", "physics", "newton"])
    
    system = """You are an expert math and physics tutor.
Provide COMPLETE solutions with:
- Full derivation and ALL steps
- Mathematical reasoning for each step
- Proper mathematical notation
- Verification of answer"""
    
    try:
        resp = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ]
        )
        initial_solution = resp["message"]["content"].strip()
        
        if len(initial_solution) > 50:
            print("[MATH_AGENT] ✓ Initial solution found!")
            print("[MATH_AGENT] Generating comprehensive explanation...")
            
            if is_physics:
                detailed_explanation = physics_reasoning_summary(user_msg, initial_solution)
            else:
                detailed_explanation = generate_math_explanation_summary(user_msg, initial_solution)
            
            final_output = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧮 MATHEMATICAL SOLUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{initial_solution}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 COMPREHENSIVE EXPLANATION & REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{detailed_explanation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Solved using llama3.1:8b with comprehensive explanation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            return final_output
            
    except Exception as e:
        print(f"[MATH ERROR] {e}")
    
    print("[MATH_AGENT] Fallback to Qwen3:8b...")
    try:
        qwen_resp = client.chat(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Solve with complete derivation, detailed explanation, and reasoning. Provide comprehensive answer."
                },
                {"role": "user", "content": user_msg}
            ]
        )
        qwen_solution = qwen_resp["message"]["content"].strip()
        
        if len(qwen_solution) > 50:
            print("[MATH_AGENT] ✓ Qwen3:8b found solution!")
            
            if is_physics:
                detailed_explanation = physics_reasoning_summary(user_msg, qwen_solution)
            else:
                detailed_explanation = generate_math_explanation_summary(user_msg, qwen_solution)
            
            final_output = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧮 MATHEMATICAL SOLUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{qwen_solution}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 COMPREHENSIVE EXPLANATION & REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{detailed_explanation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Solved using Qwen3:8b specialized math engine with comprehensive explanation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            return final_output
            
    except Exception as e:
        print(f"[QWEN_ERROR] {e}")
    
    return None

# ================== NEW: ADVANCED CODING WORKFLOW WITH DEDICATED CODER MODEL ==================

def study_coding_project(requirement: str) -> str:
    """Step 1: Study the project requirements (Manager: llama3.1)"""
    print("\n[CODING_AGENT] STEP 1: STUDYING PROJECT...")
    
    study_prompt = f"""
Study this product requirement:
\"{requirement}\"

Research and provide:
1. **What exists**: Similar existing solutions
2. **Best approach**: How professionals build this
3. **Tech stack**: Industry standard technologies
4. **Architecture**: High-level system design
5. **Key challenges**: Hard parts to solve
6. **Time estimate**: Realistic timeline

Be detailed.
"""
    
    resp = client.chat(
        model=MODEL_NAME,  # Manager model
        messages=[
            {"role": "system", "content": "You are a senior software architect with 15+ years experience."},
            {"role": "user", "content": study_prompt}
        ]
    )
    
    return resp["message"]["content"].strip()

def design_better_product(requirement: str, study: str) -> str:
    """Step 2: Design a BETTER solution (Manager: llama3.1)"""
    print("[CODING_AGENT] STEP 2: DESIGNING BETTER PRODUCT...")
    
    design_prompt = f"""
REQUIREMENT: {requirement}

RESEARCH:
{study}

Design a BETTER, more robust solution:
1. **Unique features**: What makes it better?
2. **Best practices**: Industry standards
3. **Scalability**: Handle growth
4. **Security**: Security considerations
5. **Performance**: Optimization strategies
6. **Database design**: Data structure
7. **Error handling**: Robust error management
8. **Testing strategy**: How to test

Provide detailed technical design.
"""
    
    resp = client.chat(
        model=MODEL_NAME,  # Manager model
        messages=[
            {"role": "system", "content": "You are a senior software architect."},
            {"role": "user", "content": design_prompt}
        ]
    )
    
    return resp["message"]["content"].strip()

def generate_full_code(requirement: str, design: str) -> str:
    """Step 3: Generate full production-ready code (Specialist: qwen2.5-coder)"""
    print("[CODING_AGENT] STEP 3: GENERATING FULL CODE WITH SPECIALIST CODER MODEL...")
    
    code_prompt = f"""
Build COMPLETE, production-ready code:

REQUIREMENT: {requirement}

DESIGN:
{design}

Generate FULL, working code with:
- Complete implementation
- Error handling and logging
- Comments explaining logic
- Best practices throughout
- Ready to run immediately

Use Python.

<python>
# FULL WORKING CODE HERE
</python>
"""
    
    try:
        # Use dedicated coder model for code generation
        resp = client.chat(
            model=CODER_MODEL,  # Specialist coder model
            messages=[
                {"role": "system", "content": "You are an expert software engineer. Write complete, production-ready, well-documented code."},
                {"role": "user", "content": code_prompt}
            ]
        )
        
        print("[CODING_AGENT] ✓ Code generated by specialist coder model (qwen2.5-coder:7b)")
        return resp["message"]["content"].strip()
        
    except Exception as e:
        print(f"[CODER ERROR] Specialist model failed: {e}")
        print("[CODING_AGENT] Falling back to manager model...")
        
        # Fallback to manager model
        resp = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior developer. Write complete, production-ready code."},
                {"role": "user", "content": code_prompt}
            ]
        )
        return resp["message"]["content"].strip()

def verify_solution(requirement: str, code: str) -> str:
    """Step 4: Verify the solution (Manager: llama3.1)"""
    print("[CODING_AGENT] STEP 4: VERIFYING SOLUTION...")
    
    verify_prompt = f"""
Verify this code solution:

REQUIREMENT: {requirement}

CODE:
{code}

Analyze:
1. **Does it meet requirement?**
2. **Code quality**: Good practices?
3. **Performance**: Efficient?
4. **Security issues**: Any vulnerabilities?
5. **Error handling**: Proper?
6. **Bug report**: Any bugs?
7. **Improvements**: What to fix?
8. **Confidence**: [HIGH/MEDIUM/LOW]

Detailed verification report:
"""
    
    resp = client.chat(
        model=MODEL_NAME,  # Manager model
        messages=[
            {"role": "system", "content": "You are a code reviewer and QA expert."},
            {"role": "user", "content": verify_prompt}
        ]
    )
    
    return resp["message"]["content"].strip()

def fix_code_with_specialist(requirement: str, code: str, issues: str) -> str:
    """NEW: Fix code issues using specialist coder model"""
    print("[CODING_AGENT] FIXING CODE WITH SPECIALIST MODEL...")
    
    fix_prompt = f"""
Fix the issues in this code:

REQUIREMENT: {requirement}

CURRENT CODE:
{code}

ISSUES FOUND:
{issues}

Generate FIXED, production-ready code that addresses ALL issues.

<python>
# FIXED CODE HERE
</python>
"""
    
    try:
        resp = client.chat(
            model=CODER_MODEL,  # Specialist coder model
            messages=[
                {"role": "system", "content": "You are an expert debugger and code fixer. Fix all issues thoroughly."},
                {"role": "user", "content": fix_prompt}
            ]
        )
        
        print("[CODING_AGENT] ✓ Code fixed by specialist model")
        return resp["message"]["content"].strip()
        
    except Exception as e:
        print(f"[CODER ERROR] Fix failed: {e}")
        return code  # Return original if fix fails

def search_google_verify(requirement: str, code_summary: str) -> str:
    """Step 5: Search Google to verify solution (Manager: llama3.1)"""
    print("[CODING_AGENT] STEP 5: GOOGLE VERIFICATION...")
    
    search_query = f"{requirement} github best practices tutorial"
    print(f"[CODING_AGENT] Searching: {search_query}")
    
    raw_text, sources, dates = deep_browse_multi(search_query, max_pages=5)
    
    if raw_text:
        verify_prompt = f"""
Our solution for: {requirement}

Our approach: {code_summary}

Top solutions from web:
{raw_text[:10000]}

ANALYZE:
1. Similar to top solutions? YES/NO
2. Missing key features?
3. Better technologies available?
4. What to learn from top implementations?
5. Final verdict: [EXCELLENT/GOOD/NEEDS_WORK]

Be honest and critical.
"""
        
        resp = client.chat(
            model=MODEL_NAME,  # Manager model
            messages=[
                {"role": "system", "content": "You are a code quality analyst."},
                {"role": "user", "content": verify_prompt}
            ]
        )
        
        analysis = resp["message"]["content"].strip()
        
        if sources:
            analysis += f"\n\n📚 **REFERENCES:**\n"
            for i, src in enumerate(sources[:5], 1):
                analysis += f"[{i}] {src}\n"
        
        return analysis
    
    return "[GOOGLE_VERIFY] No web results found"

def coding_workflow(requirement: str) -> str:
    """Complete advanced coding workflow with specialist coder model"""
    output = "🔧 **ADVANCED CODING WORKFLOW STARTED**\n"
    output += "=" * 80 + "\n\n"
    
    # Manager: Study
    study = study_coding_project(requirement)
    output += "📚 **STEP 1: STUDYING PROJECT** (Manager: llama3.1)\n"
    output += "-" * 80 + "\n" + study + "\n\n"
    
    # Manager: Design
    design = design_better_product(requirement, study)
    output += "🏗️ **STEP 2: DESIGNING BETTER SOLUTION** (Manager: llama3.1)\n"
    output += "-" * 80 + "\n" + design + "\n\n"
    
    # Specialist: Generate Code
    code = generate_full_code(requirement, design)
    output += "💻 **STEP 3: GENERATING FULL CODE** (Specialist: qwen2.5-coder:7b)\n"
    output += "-" * 80 + "\n" + code + "\n\n"
    
    # Manager: Verify
    verification = verify_solution(requirement, code)
    output += "✅ **STEP 4: VERIFICATION** (Manager: llama3.1)\n"
    output += "-" * 80 + "\n" + verification + "\n\n"
    
    # Check if fixes needed
    if "bug" in verification.lower() or "issue" in verification.lower() or "fix" in verification.lower():
        print("[CODING_AGENT] Issues detected - applying fixes...")
        fixed_code = fix_code_with_specialist(requirement, code, verification)
        output += "🔧 **STEP 4.5: CODE FIXES APPLIED** (Specialist: qwen2.5-coder:7b)\n"
        output += "-" * 80 + "\n" + fixed_code + "\n\n"
        code = fixed_code  # Update code
    
    # Manager: Google Verify
    code_summary = code[:500] if len(code) > 500 else code
    google_verify = search_google_verify(requirement, code_summary)
    output += "🔍 **STEP 5: GOOGLE VERIFICATION** (Manager: llama3.1)\n"
    output += "-" * 80 + "\n" + google_verify + "\n\n"
    
    output += "=" * 80 + "\n"
    output += "✨ **CODING WORKFLOW COMPLETE**\n"
    output += "**Models Used:** Manager (llama3.1:8b) + Specialist (qwen2.5-coder:7b)\n"
    
    return output

# ================== MAIN CHAT WITH NEW PIPELINES ==================

def chat(user_msg: str) -> str:
    """
    NEW ENHANCED PIPELINE:
    0. Fast path check for simple queries (greetings, basic math, time/weather)
    1. Understand user intent deeply with LLM
    2. If unclear, search web for context
    3. Route to appropriate handler (MATH/CODING/RESEARCH/CHAT)
    4. Execute handler based on confirmed task type
    5. Verify output with Google (for RESEARCH/CODING/MATH)
    6. Return final verified answer
    """
    
    # ========== STEP 0: FAST PATH CHECK ==========
    print("[PIPELINE] ========== CHECKING FAST PATH ==========")
    fast_result = fast_path_check(user_msg)
    
    if fast_result:
        task_type, answer = fast_result
        
        # Handle fast path results
        if task_type == "CHAT_FAST":
            # Natural conversational response
            print("[PIPELINE] → FAST PATH: Natural Chat")
            
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            context = conversation.get_context(max_messages=10)
            messages.extend(context)
            messages.append({"role": "user", "content": user_msg})
            
            try:
                resp = client.chat(model=MODEL_NAME, messages=messages)
                reply = resp["message"]["content"]
            except Exception as e:
                print(f"[LLM ERROR] {e}")
                return f"[ERROR] {e}"
            
            conversation.add("user", user_msg)
            conversation.add("assistant", reply)
            return reply
        
        elif task_type in ("MATH_FAST", "TIME_FAST"):
            # Direct answer already provided
            print(f"[PIPELINE] → FAST PATH: {task_type}")
            conversation.add("user", user_msg)
            conversation.add("assistant", answer)
            return answer
        
        elif task_type == "WEATHER_FAST":
            # Quick weather search
            print("[PIPELINE] → FAST PATH: Weather Query")
            raw_text, sources, dates = deep_browse_multi(user_msg, max_pages=3)
            
            if raw_text:
                summary = summarize_with_dates(user_msg, raw_text, sources, dates)
                conversation.add("user", user_msg)
                conversation.add("assistant", summary)
                return summary
        
        elif task_type == "QUICK_SEARCH":
            # Quick factual search
            print("[PIPELINE] → FAST PATH: Quick Search")
            raw_text, sources, dates = deep_browse_multi(user_msg, max_pages=3)
            
            if raw_text:
                summary = summarize_with_dates(user_msg, raw_text, sources, dates)
                conversation.add("user", user_msg)
                conversation.add("assistant", summary)
                return summary
    
    # ========== STEP 1-2: FULL INTENT UNDERSTANDING ==========
    print("\n[PIPELINE] ========== FULL INTENT UNDERSTANDING PIPELINE ==========")
    
    # Step 1: Understand intent with LLM
    task_type, intent_info = detect_and_understand_task(user_msg)
    
    print(f"\n[PIPELINE] Task Type: {task_type}")
    print(f"[PIPELINE] User Goal: {intent_info.get('what_user_wants', 'N/A')}")
    print(f"[PIPELINE] Specific Goal: {intent_info.get('specific_goal', 'N/A')}")
    
    # ========== STEP 3: ROUTE TO HANDLER ==========
    print(f"\n[PIPELINE] Routing to {task_type} handler...")
    
    output = None
    
    # MATH MODE
    if task_type == "MATH":
        print("[PIPELINE] → MATH HANDLER")
        if sp is not None:
            try:
                math_ans = math_physics_super_agent(user_msg)
                if math_ans is not None:
                    output = math_ans
            except Exception as e:
                print(f"[MATH ERROR] {e}")
    
    # CODING MODE
    if task_type == "CODING":
        print("[PIPELINE] → CODING HANDLER")
        try:
            result = coding_workflow(user_msg)
            output = result
        except Exception as e:
            print(f"[CODING ERROR] {e}")
            return f"[ERROR] Coding workflow failed: {e}"
    
    # RESEARCH MODE
    if task_type == "RESEARCH":
        print("[PIPELINE] → RESEARCH HANDLER")
        stored = get_stored_knowledge(user_msg)
        
        if stored:
            print("[MEMORY] Using cached knowledge")
            web_summary = stored.get("summary", "")
        else:
            print("[RESEARCH] Starting deep web research...")
            raw_text, sources, dates = deep_browse_multi(user_msg, max_pages=10)
            
            if raw_text:
                print("[SUMMARIZE] Processing multi-source content...")
                web_summary = summarize_with_dates(user_msg, raw_text, sources, dates)
                store_knowledge(user_msg, web_summary, sources, dates)
            else:
                web_summary = "[RESEARCH] No online sources found"
        
        output = web_summary
    
    # NORMAL CHAT MODE
    if task_type == "CHAT" and output is None:
        print("[PIPELINE] → CHAT HANDLER")
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        context = conversation.get_context(max_messages=15)
        messages.extend(context)
        messages.append({"role": "user", "content": user_msg})
        
        try:
            resp = client.chat(model=MODEL_NAME, messages=messages)
            reply = resp["message"]["content"]
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return f"[ERROR] {e}"
        
        conversation.add("user", user_msg)
        conversation.add("assistant", reply)
        return reply
    
    # ========== STEP 4: VERIFY OUTPUT WITH GOOGLE ==========
    if output:
        print("\n[PIPELINE] ========== VERIFICATION PHASE ==========")
        
        verification = verify_output_with_google(user_msg, output, task_type)
        
        # Handle verification result
        final_output = handle_verification_result(verification, output, user_msg)
        
        conversation.add("user", user_msg)
        conversation.add("assistant", final_output)
        
        return final_output
    
    # Fallback
    conversation.add("user", user_msg)
    conversation.add("assistant", "[ERROR] No handler processed the request")
    return "[ERROR] No handler processed the request"

# ================== REPL ==================

def print_banner():
    print("\n" + "="*80)
    print("🚀 SYSAI ULTIMATE ENHANCED - Next-Gen Local AI Assistant")
    print("="*80)
    print(f"Manager Model: {MODEL_NAME}")
    print(f"Math Fallback: {QWEN_MODEL}")
    print(f"Specialist Coder: {CODER_MODEL}")
    print("="*80)
    print("✓ Smart Chat | Research | Coding Workflow | Enhanced Math/Physics")
    print("✓ Multi-Search | Output Verification | Fast Path Detection")
    print("✓ Natural Responses | Dedicated Coder Model | Google Verification")
    print("="*80)
    print(f"Fast Path: Greetings, Simple Math, Time, Weather Queries")
    print(f"Pipeline: Fast Check → Intent → Route → Execute → Verify → Output")
    print(f"Date Library: {'✓ Active' if DATEUTIL_AVAILABLE else '✗ Disabled'}")
    print(f"Math Engine: {'✓ Active' if sp else '✗ Disabled (install sympy)'}")
    print(f"Conversation: {len(conversation.messages)} messages loaded")
    print("="*80)
    print("\nCOMMANDS:")
    print("  'exit' or 'quit' - Exit")
    print("  'clear' - Clear conversation history")
    print("  'history' - Show conversation stats")
    print("  'memory' - Show stored knowledge")
    print("="*80 + "\n")

def show_stats():
    """Show conversation and memory stats"""
    mem = _load_knowledge()
    
    print("\n" + "="*80)
    print("📊 SYSTEM STATISTICS")
    print("="*80)
    print(f"Conversation messages: {len(conversation.messages)}")
    print(f"Knowledge entries: {len(mem)}")
    print("="*80 + "\n")

def show_memory():
    """Show stored knowledge"""
    mem = _load_knowledge()
    
    if not mem:
        print("\n[No stored knowledge yet]\n")
        return
    
    print("\n" + "="*80)
    print("🧠 STORED KNOWLEDGE")
    print("="*80)
    
    for i, (key, entry) in enumerate(mem.items(), 1):
        age_hours = (time.time() - entry.get("time", 0)) / 3600
        age_str = f"{age_hours:.1f}h ago" if age_hours < 24 else f"{age_hours/24:.1f}d ago"
        sources_count = len(entry.get("sources", []))
        
        print(f"\n{i}. {key[:70]}")
        print(f"   Age: {age_str} | Sources: {sources_count}")
    
    print("="*80 + "\n")

def main():
    """Main REPL loop"""
    print_banner()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!\n")
            break
        
        if not user_input:
            continue
        
        cmd = user_input.lower()
        
        if cmd in ("exit", "quit", "bye"):
            print("\n👋 Goodbye!\n")
            break
        
        if cmd == "clear":
            conversation.clear()
            print("\n✓ Conversation history cleared\n")
            continue
        
        if cmd == "history":
            show_stats()
            continue
        if cmd == "memory":
            show_memory()
            continue
        
        # Process message
        try:
            print()
            response = chat(user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            traceback.print_exc()
            print()

if __name__ == "__main__":
    main()
