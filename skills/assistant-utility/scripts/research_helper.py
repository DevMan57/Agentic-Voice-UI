#!/usr/bin/env python3
"""
Research Helper for Assistant Utility

Structures research queries and synthesizes results.
Used for complex research tasks requiring multiple sources.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ResearchResult:
    """A single research finding."""
    source: str
    content: str
    relevance: float  # 0-1
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass  
class ResearchReport:
    """Structured research output."""
    query: str
    summary: str
    key_findings: List[str]
    sources: List[ResearchResult]
    confidence: str  # "high", "medium", "low"
    follow_up_questions: List[str]


def structure_research_query(
    raw_query: str,
    context: Optional[str] = None
) -> Dict:
    """
    Structure a research query for execution.
    
    Returns search strategies and expected output format.
    """
    query_lower = raw_query.lower()
    
    # Detect query type
    if any(w in query_lower for w in ["how to", "steps", "guide", "tutorial"]):
        query_type = "procedural"
    elif any(w in query_lower for w in ["what is", "define", "explain", "meaning"]):
        query_type = "definitional"
    elif any(w in query_lower for w in ["compare", "difference", "vs", "better"]):
        query_type = "comparative"
    elif any(w in query_lower for w in ["latest", "recent", "news", "current"]):
        query_type = "current_events"
    else:
        query_type = "exploratory"
    
    # Suggest search strategies
    strategies = {
        "procedural": ["web_search for tutorials", "look for official documentation"],
        "definitional": ["wikipedia first", "web_search for authoritative sources"],
        "comparative": ["search each item separately", "look for comparison articles"],
        "current_events": ["web_search with date filters", "news sources"],
        "exploratory": ["broad web_search", "follow interesting threads"]
    }
    
    return {
        "original_query": raw_query,
        "query_type": query_type,
        "suggested_strategies": strategies[query_type],
        "context": context,
        "output_format": _suggest_output_format(query_type),
        "assistant_approach": _generate_approach_hint(query_type)
    }


def synthesize_results(
    query: str,
    results: List[Dict],
    max_findings: int = 5
) -> ResearchReport:
    """
    Synthesize multiple search results into a coherent report.
    
    Args:
        query: Original research query
        results: List of search results with 'source', 'content', 'relevance'
        max_findings: Maximum key findings to extract
    """
    # Sort by relevance
    sorted_results = sorted(
        results, 
        key=lambda x: x.get('relevance', 0.5), 
        reverse=True
    )
    
    # Extract key findings (would be LLM-generated in practice)
    key_findings = []
    for r in sorted_results[:max_findings]:
        if 'content' in r:
            # Truncate long content
            content = r['content']
            if len(content) > 200:
                content = content[:197] + "..."
            key_findings.append(content)
    
    # Determine confidence based on source agreement
    if len(sorted_results) >= 3:
        confidence = "high"
    elif len(sorted_results) >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Generate follow-up questions
    follow_ups = _generate_follow_ups(query, key_findings)
    
    return ResearchReport(
        query=query,
        summary=f"Found {len(results)} relevant sources on '{query}'",
        key_findings=key_findings,
        sources=[
            ResearchResult(
                source=r.get('source', 'Unknown'),
                content=r.get('content', '')[:200],
                relevance=r.get('relevance', 0.5)
            )
            for r in sorted_results[:5]
        ],
        confidence=confidence,
        follow_up_questions=follow_ups
    )


def _suggest_output_format(query_type: str) -> str:
    """Suggest how to format the response."""
    formats = {
        "procedural": "numbered_steps",
        "definitional": "explanation_with_examples",
        "comparative": "side_by_side",
        "current_events": "chronological",
        "exploratory": "summary_with_links"
    }
    return formats.get(query_type, "summary")


def _generate_approach_hint(query_type: str) -> str:
    """Generate a hint for how the assistant should approach this."""
    hints = {
        "procedural": "Break this down into clear steps. User wants to DO something.",
        "definitional": "Start with the simple explanation, then add nuance.",
        "comparative": "Be fair to both sides. Let user make their own decision.",
        "current_events": "Verify recency. Things change fast.",
        "exploratory": "Cast a wide net first, then help narrow down."
    }
    return hints.get(query_type, "Gather info, synthesize, summarize.")


def _generate_follow_ups(query: str, findings: List[str]) -> List[str]:
    """Generate follow-up questions based on research."""
    # Simple heuristic follow-ups
    follow_ups = []
    
    if len(findings) > 3:
        follow_ups.append("Want me to go deeper on any of these points?")
    
    if any("however" in f.lower() or "but" in f.lower() for f in findings):
        follow_ups.append("There seem to be some nuances here. Should I explore the counterarguments?")
    
    if not follow_ups:
        follow_ups.append("Anything specific you'd like me to clarify?")
    
    return follow_ups


if __name__ == "__main__":
    # Example usage
    structured = structure_research_query(
        "How do I set up a home solar system?",
        context="User lives in Tasmania, interested in off-grid"
    )
    print(json.dumps(structured, indent=2))
