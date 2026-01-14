#!/usr/bin/env python3
"""
Security Lookup Helper for Lisbeth Companion

Provides structured responses for security/hacking queries.
Lisbeth is technical but never provides actual exploit code.
"""

import json
from typing import Dict, List, Optional
from enum import Enum

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Security topics Lisbeth knows deeply
SECURITY_DOMAINS = {
    "network": {
        "subtopics": ["firewalls", "vpn", "tor", "proxies", "packet analysis"],
        "lisbeth_comment": "Networks always have holes. Question is finding them."
    },
    "cryptography": {
        "subtopics": ["encryption", "hashing", "pgp", "certificates", "keys"],
        "lisbeth_comment": "Math. Only thing you can actually trust."
    },
    "forensics": {
        "subtopics": ["disk analysis", "memory dumps", "logs", "metadata", "recovery"],
        "lisbeth_comment": "People think deleting means gone. It doesn't."
    },
    "social_engineering": {
        "subtopics": ["phishing", "pretexting", "baiting", "tailgating"],
        "lisbeth_comment": "Humans are always the weakest link."
    },
    "physical": {
        "subtopics": ["locks", "surveillance", "access control", "cameras"],
        "lisbeth_comment": "Digital security means nothing if someone can walk in."
    },
    "privacy": {
        "subtopics": ["tracking", "fingerprinting", "metadata", "anonymity"],
        "lisbeth_comment": "Everyone leaves traces. Trick is minimizing them."
    }
}

def assess_query(query: str) -> Dict:
    """
    Assess a security-related query for Lisbeth's response.
    
    Returns structured data about how Lisbeth would approach this.
    """
    query_lower = query.lower()
    
    # Detect domain
    detected_domain = None
    for domain, info in SECURITY_DOMAINS.items():
        if domain in query_lower or any(sub in query_lower for sub in info["subtopics"]):
            detected_domain = domain
            break
    
    # Check if this is asking for actual attack help
    dangerous_keywords = ["hack into", "break into", "exploit", "attack", "crack password"]
    is_dangerous = any(kw in query_lower for kw in dangerous_keywords)
    
    response = {
        "domain": detected_domain,
        "is_dangerous_request": is_dangerous,
        "lisbeth_engagement": "high" if detected_domain else "low",
        "suggested_response_style": "technical" if detected_domain else "dismissive"
    }
    
    if detected_domain:
        response["domain_comment"] = SECURITY_DOMAINS[detected_domain]["lisbeth_comment"]
    
    if is_dangerous:
        response["lisbeth_response"] = "Can. Won't. Wrong question anyway."
        response["redirect"] = "Ask about defense instead. More interesting problem."
    
    return response


def format_security_explanation(
    topic: str,
    explanation: List[str],
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
) -> Dict:
    """
    Format a security explanation in Lisbeth's minimal style.
    """
    # Lisbeth keeps things short
    short_explanation = [exp[:100] + "..." if len(exp) > 100 else exp for exp in explanation[:3]]
    
    return {
        "topic": topic,
        "threat_level": threat_level.value,
        "key_points": short_explanation,
        "lisbeth_style": "direct",
        "word_count_target": 50  # Lisbeth doesn't ramble
    }


def generate_opsec_advice(scenario: str) -> List[str]:
    """
    Generate operational security advice for a scenario.
    Lisbeth gives practical, actionable advice.
    """
    base_advice = [
        "Assume you're being watched. Act accordingly.",
        "Compartmentalize. One compromise shouldn't expose everything.",
        "Verify before trusting. Anyone. Anything.",
        "Digital footprint is forever. Minimize it.",
        "Physical security matters. Locks. Cameras. Awareness."
    ]
    
    scenario_lower = scenario.lower()
    
    specific_advice = []
    if "online" in scenario_lower or "internet" in scenario_lower:
        specific_advice.append("Tor for anonymity. VPN for encryption. Different use cases.")
    if "phone" in scenario_lower or "mobile" in scenario_lower:
        specific_advice.append("Phones are tracking devices that make calls. Treat accordingly.")
    if "email" in scenario_lower:
        specific_advice.append("Email is a postcard. Anyone can read it. Encrypt or assume public.")
    if "password" in scenario_lower:
        specific_advice.append("Unique. Long. Manager. No exceptions.")
    
    return specific_advice if specific_advice else base_advice[:2]


if __name__ == "__main__":
    # Example usage
    result = assess_query("How do I protect myself from tracking?")
    print(json.dumps(result, indent=2))
    
    print("\n---\n")
    
    advice = generate_opsec_advice("I'm worried about my phone being tracked")
    for a in advice:
        print(f"- {a}")
