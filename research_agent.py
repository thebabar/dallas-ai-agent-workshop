"""Research agent using Tavily for web search and LangGraph for orchestration."""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from openai import OpenAI
import os

# ==================== STATE ====================
class ResearchState(TypedDict):
    question: str
    search_queries: List[str]
    raw_results: List[dict]
    report: str

# ==================== CLIENTS ====================
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", "").strip())

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
    default_headers={
        "HTTP-Referer": "http://localhost:8888",
        "X-Title": "Dallas Agent Workshop - Research",
    },
)

MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")

# ==================== NODES ====================

def planner(state: ResearchState) -> ResearchState:
    """Generate 2-4 search queries."""
    prompt = f"""You are a research planner. Break this question into 2-4 specific search queries.

Question: {state['question']}

Return ONLY a Python list of strings. Example: ["query 1", "query 2"]"""
    
    resp = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    queries_str = resp.choices[0].message.content.strip()
    # Extract list from response (handles markdown code blocks)
    if "```" in queries_str:
        queries_str = queries_str.split("```")[1].replace("python", "").strip()
    
    queries = eval(queries_str)
    
    print(f"\n{'='*60}")
    print("PLANNED QUERIES:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print(f"{'='*60}\n")
    
    state["search_queries"] = queries
    return state


def searcher(state: ResearchState) -> ResearchState:
    """Execute Tavily searches."""
    all_results = []
    
    for query in state["search_queries"]:
        print(f"🔍 Searching: {query}")
        try:
            response = tavily_client.search(
                query=query,
                max_results=3,
                search_depth="basic"
            )
            results = response.get("results", [])
            all_results.extend(results)
            print(f"   → Found {len(results)} sources")
        except Exception as e:
            print(f"   ⚠️  Search failed: {e}")
    
    print(f"\n✓ Collected {len(all_results)} total sources\n")
    
    state["raw_results"] = all_results
    return state


def synthesizer(state: ResearchState) -> ResearchState:
    """Generate structured report from sources."""
    
    sources_text = "\n\n".join([
        f"SOURCE [{i+1}]:\nTitle: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:400]}..."
        for i, r in enumerate(state["raw_results"][:10])
    ])
    
    prompt = f"""You are a research analyst. Write a clear, structured report answering this question using ONLY the sources below.

QUESTION: {state['question']}

SOURCES:
{sources_text}

FORMAT:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points with citations like [1], [2])
3. Conclusion (1-2 sentences)

Keep under 400 words. Cite every claim. If sources don't fully answer the question, say so.

REPORT:"""
    
    resp = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    
    report = resp.choices[0].message.content
    state["report"] = report
    return state


# ==================== GRAPH ====================

def build_research_graph():
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("planner", planner)
    workflow.add_node("searcher", searcher)
    workflow.add_node("synthesizer", synthesizer)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# ==================== PUBLIC API ====================

def run_research(question: str) -> dict:
    """Run the research agent and return final state."""
    
    initial_state = {
        "question": question,
        "search_queries": [],
        "raw_results": [],
        "report": "",
    }
    
    graph = build_research_graph()
    result = graph.invoke(initial_state)
    
    return result
```
