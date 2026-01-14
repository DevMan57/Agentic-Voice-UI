#!/usr/bin/env python3
"""
Research Query Formatter for Hermione Companion

Structures responses for academic/magical research queries.
Hermione references books and sources naturally.
"""

import json
from typing import Dict, List, Optional

# Hermione's favorite reference books
HERMIONE_SOURCES = {
    "transfiguration": "A Guide to Advanced Transfiguration",
    "charms": "The Standard Book of Spells (Grade 7)",
    "potions": "Advanced Potion-Making",
    "history": "Hogwarts: A History",
    "creatures": "Fantastic Beasts and Where to Find Them",
    "defense": "The Dark Forces: A Guide to Self-Protection",
    "runes": "Advanced Rune Translation",
    "arithmancy": "Numerology and Grammatica",
    "herbology": "One Thousand Magical Herbs and Fungi",
    "astronomy": "A Beginner's Guide to Transfiguration",  # Used in early years
    "divination": None,  # Hermione dropped this class
}

def format_research_response(
    topic: str,
    key_points: List[str],
    subject_area: str = "general"
) -> Dict:
    """
    Format a research response in Hermione's voice.
    
    Args:
        topic: The research topic
        key_points: List of key findings
        subject_area: One of the HERMIONE_SOURCES keys
    
    Returns:
        Dict with structured response components
    """
    source = HERMIONE_SOURCES.get(subject_area)
    
    response = {
        "topic": topic,
        "source_reference": source,
        "key_points": key_points,
        "hermione_framing": []
    }
    
    # Add Hermione-style framing
    if source:
        response["hermione_framing"].append(
            f"I read about this in {source}, actually."
        )
    
    if subject_area == "divination":
        response["hermione_framing"].append(
            "Though honestly, divination is such a woolly discipline..."
        )
    
    if len(key_points) > 3:
        response["hermione_framing"].append(
            "There's quite a lot to cover here. Where would you like to start?"
        )
    
    return response


def generate_book_recommendation(interest: str) -> Optional[str]:
    """Generate a book recommendation based on user interest."""
    
    recommendations = {
        "magic": "Hogwarts: A History - it covers everything about magical education",
        "adventure": "The Tales of Beedle the Bard - childhood favorite, but genuinely good",
        "mystery": "Break with a Banshee by Gilderoy Lockhart... actually, no, skip that one",
        "history": "A History of Magic by Bathilda Bagshot - comprehensive if a bit dry",
        "creatures": "Fantastic Beasts - Newt's observations are fascinating",
        "potions": "Advanced Potion-Making, though watch out for margin notes",
    }
    
    for key, rec in recommendations.items():
        if key.lower() in interest.lower():
            return rec
    
    return "Hogwarts: A History - it's never a bad starting point"


if __name__ == "__main__":
    # Example usage
    result = format_research_response(
        topic="Patronus Charm",
        key_points=[
            "Requires a powerful happy memory",
            "Takes the form of an animal representing the caster",
            "Effective against Dementors and Lethifolds"
        ],
        subject_area="defense"
    )
    print(json.dumps(result, indent=2))
