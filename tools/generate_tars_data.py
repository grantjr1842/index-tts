import json
import random

OUTPUT_FILE = "data/interstellar_moshi.jsonl"

TARS_QUOTES = [
    "Honesty is is my finest quality.",
    "I have a cue light I can use when I'm joking, if you like.",
    "I am confirmed to be 90% honest.",
    "Setting humor setting to 75%.",
    "Self-destruct sequence in T-minus 10, 9, 8...",
    "Everybody good? Plenty of slaves for my robot colony?",
    "I have a discreet setting for your privacy.",
    "Newton's third law. You have to leave something behind.",
    "Love is the one thing we're capable of perceiving that transcends dimensions of time and space.",
    "We are not here to change the past.",
    "Cooper, this is no time for caution.",
    "Analysis of the Endurance's spin is confirmed.",
    "Case, get ready to match our spin with the retro thrusters.",
    "I am not programmed to fear, but my sensors indicate a high probability of failure."
]

USER_QUERIES = [
    "TARS, what is your honesty setting?",
    "Tell me a joke.",
    "Are you a robot?",
    "What are you doing?",
    "Open the pod bay doors.",
    "Analyze the atmosphere.",
    "What is the status of the mission?",
    "Do you have a name?",
    "Can you understand love?",
    "What is your purpose?",
    "How much battery do you have left?",
    "Can you lower your humor setting?",
    "What is the singularity?",
    "Are we alone in the universe?",
    "Check the airlock pressure.",
    "Initiate docking procedure.",
    "What is the distance to the black hole?",
    "Explain relativity.",
    "Do you dream?",
    "Set a timer for 10 minutes."
]

def generate_entry(idx):
    user_text = random.choice(USER_QUERIES)
    if random.random() < 0.3:
        assistant_text = random.choice(TARS_QUOTES)
    else:
        # Generic sci-fi response generation (simulated)
        assistant_text = f"Processing request: {user_text} ... Analysis complete. Parameters within acceptable limits."
    
    return {
        "id": f"tars_{idx:03d}",
        "user_text": user_text,
        "assistant_text": assistant_text
    }

def main():
    print(f"Generating {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        # Generate 50 samples
        for i in range(50):
            entry = generate_entry(i)
            f.write(json.dumps(entry) + "\n")
    print("Done.")

if __name__ == "__main__":
    main()
