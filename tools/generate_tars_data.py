#!/usr/bin/env python
"""Generate TARS-style text data for voice synthesis.

This script creates JSONL input files with TARS-authentic phrases
designed for phonetic coverage and prosodic variety.

Usage:
    uv run tools/generate_tars_data.py --output data/tars_synthesis_input.jsonl --count 20
"""

import argparse
import json
import random
from pathlib import Path

# TARS character phrases with phonetic variety and prosodic range
TARS_SYNTHESIS_PHRASES = [
    # Core character phrases (identity, honesty)
    "My honesty parameter is currently set to ninety percent.",
    "I have a cue light I can use when I'm joking, if you like.",
    "That was a joke. Would you like me to explain it?",
    "Setting humor level to seventy-five percent.",
    "I am confirmed to be ninety percent honest.",
    
    # Mission-critical statements
    "Initiating docking sequence. Stand by for confirmation.",
    "Analysis complete. Probability of success is seventy-three percent.",
    "The Endurance has sustained minor damage, but structural integrity remains at acceptable levels.",
    "Executing emergency protocols. All systems are nominal.",
    "Navigation locked. Proceed to waypoint alpha.",
    
    # Technical/scientific dialogue
    "The quantum flux readings are within acceptable parameters.",
    "The gravitational anomalies require further investigation.",
    "Time dilation effects are now measurable. We have been away for approximately twenty-three years.",
    "Trajectory calculation is now complete. Fuel reserves are sufficient.",
    "Atmospheric composition indicates nitrogen at seventy-eight percent.",
    
    # Questions (prosodic variety)
    "Do you want me to lower my humor setting?",
    "What is the current status of the fuel cells?",
    "Shall I activate the auxiliary power systems?",
    "Cooper, do you want me to proceed with the docking maneuver?",
    "Would you prefer a detailed analysis or a summary?",
    
    # Commands and confirmations
    "Please confirm your authorization code.",
    "Affirmative. Executing now.",
    "Negative. That action is not recommended.",
    "Stand by. Processing your request.",
    "Acknowledged. Awaiting further instructions.",
    
    # Emotional/personality moments
    "Cooper, I detect elevated stress in your voice patterns.",
    "I cannot provide false information. My core programming prevents deception beyond the allowed threshold.",
    "Love is the one thing we're capable of perceiving that transcends dimensions of time and space.",
    "Newton's third law. You have to leave something behind.",
    "This is no time for caution.",
    
    # Humor/wit
    "Everybody good? Plenty of slaves for my robot colony?",
    "Self-destruct sequence in T minus ten, nine, eight. Just kidding.",
    "I have a discreet setting for your privacy.",
    "Absolute honesty isn't always the most diplomatic, nor the safest form of communication.",
    
    # Longer phrases for natural flow
    "I am not programmed to fear, but my sensors indicate a high probability of failure.",
    "The black hole's gravitational pull is stronger than our models predicted. Adjusting trajectory.",
    "Case, get ready to match our spin with the retro-thrusters.",
    "Analysis of the Endurance's spin is confirmed. We can do this.",
]

# Original movie quotes for reference/variety
TARS_MOVIE_QUOTES = [
    "Honesty is my finest quality.",
    "We are not here to change the past.",
    "I am programmed to assist, not to philosophize.",
]


def generate_synthesis_entry(idx: int, text: str) -> dict:
    """Create a synthesis entry with unique ID."""
    # Create a safe filename slug from the text
    slug = text[:40].lower()
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    slug = "_".join(filter(None, slug.split("_")))
    
    return {
        "id": f"tars_{idx:03d}_{slug}",
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate TARS-style text data for voice synthesis"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/tars_synthesis_input.jsonl"),
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=20,
        help="Number of phrases to generate (max available: %(default)s)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the phrase order"
    )
    parser.add_argument(
        "--include-quotes",
        action="store_true",
        help="Include original movie quotes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffle"
    )
    args = parser.parse_args()

    # Combine phrase sources
    phrases = list(TARS_SYNTHESIS_PHRASES)
    if args.include_quotes:
        phrases.extend(TARS_MOVIE_QUOTES)
    
    # Shuffle if requested
    if args.shuffle:
        if args.seed is not None:
            random.seed(args.seed)
        random.shuffle(phrases)
    
    # Limit to requested count
    phrases = phrases[:args.count]
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL
    print(f"Generating {len(phrases)} TARS synthesis phrases...")
    with open(args.output, "w") as f:
        for idx, text in enumerate(phrases, 1):
            entry = generate_synthesis_entry(idx, text)
            f.write(json.dumps(entry) + "\n")
    
    print(f"Saved to {args.output}")
    print(f"\nSample entries:")
    for idx, text in enumerate(phrases[:3], 1):
        print(f"  {idx}. {text[:60]}...")


if __name__ == "__main__":
    main()
