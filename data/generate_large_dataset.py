"""
============================================================
generate_large_dataset.py — Build a large, diverse prompt dataset
for GAN adversarial training.

Inspired by watermark-stealing (ICML 2024) which uses:
  - C4 realnewslike (13M samples) for querying (30K queries)
  - writing-prompts, dolly-writing, essays, book reports, fake news
    for evaluation diversity

This script generates a unified prompt file with:
  1. C4 realnewslike prompts (text prefixes, like watermark-stealing)
  2. Dolly creative writing prompts
  3. Book report prompts (from watermark-stealing's generation_prompts.py)
  4. Essay/story prompts (diverse topics)
  5. Synthetic prefix prompts (news-style continuations)

Output: data/large_dataset.jsonl
  Each line: {"prompt": "...", "source": "c4|dolly|report|essay|synthetic"}

Usage:
  python scripts/generate_large_dataset.py                    # default 10K
  python scripts/generate_large_dataset.py --num_prompts 20000
  python scripts/generate_large_dataset.py --num_prompts 5000 --fast  # skip HF downloads
============================================================
"""

import os
import sys
import json
import random
import argparse
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Source 1: C4 realnewslike (main source, like watermark-stealing)
# ============================================================

def load_c4_prompts(
    num_prompts: int = 8000,
    max_prefix_tokens: int = 30,
    split: str = "train",
) -> List[dict]:
    """
    Load prompts from C4 realnewslike dataset.
    
    Watermark-stealing uses full text as input to the server,
    then the server's tokenizer truncates to fit context window.
    
    For our OPT-1.3B setup, we take the first ~30 tokens as prompt
    (matching _load_prompts in train_upv.py) so the model generates
    a continuation of ~200 tokens.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("[C4] datasets/transformers not installed. Skipping C4.")
        return []

    print(f"[C4] Loading c4/realnewslike split={split}...")
    try:
        dataset = load_dataset(
            "c4", "realnewslike",
            split=split,
            streaming=True,  # Don't download entire 13M dataset
        )
    except Exception as e:
        print(f"[C4] Failed to load: {e}")
        return []

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)
    
    prompts = []
    seen_prefixes = set()  # Deduplicate
    
    print(f"[C4] Extracting {num_prompts} prompts (prefix={max_prefix_tokens} tokens)...")
    
    for i, example in enumerate(dataset):
        if len(prompts) >= num_prompts:
            break
        
        text = example.get("text", "")
        if not text or len(text) < 50:
            continue
        
        # Tokenize and take prefix (like watermark-stealing's server truncation)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) < max_prefix_tokens + 10:
            continue
        
        # Take first max_prefix_tokens as prompt
        prefix_tokens = tokens[:max_prefix_tokens]
        prompt = tokenizer.decode(prefix_tokens, skip_special_tokens=True).strip()
        
        # Deduplicate by first 50 chars
        key = prompt[:50]
        if key in seen_prefixes:
            continue
        seen_prefixes.add(key)
        
        if len(prompt) < 20:  # Skip too-short prompts
            continue
        
        prompts.append({"prompt": prompt, "source": "c4"})
        
        if len(prompts) % 2000 == 0:
            print(f"  [{len(prompts)}/{num_prompts}]")
    
    print(f"[C4] Got {len(prompts)} prompts")
    return prompts


# ============================================================
# Source 2: Dolly creative writing
# ============================================================

def load_dolly_prompts() -> List[dict]:
    """
    Load creative writing prompts from Dolly-15K.
    Watermark-stealing uses dolly-writing-100 for evaluation.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    
    print("[Dolly] Loading databricks/databricks-dolly-15k...")
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"[Dolly] Failed: {e}")
        return []
    
    prompts = []
    for item in dataset:
        if item.get("category") == "creative_writing":
            instruction = item.get("instruction", "").strip()
            if instruction and len(instruction) > 20:
                prompts.append({"prompt": instruction, "source": "dolly"})
    
    # Also include open_qa and general_qa for diversity
    for item in dataset:
        cat = item.get("category", "")
        if cat in ("open_qa", "general_qa", "brainstorming", "summarization"):
            instruction = item.get("instruction", "").strip()
            if instruction and len(instruction) > 30:
                prompts.append({"prompt": instruction, "source": f"dolly_{cat}"})
    
    random.shuffle(prompts)
    print(f"[Dolly] Got {len(prompts)} prompts")
    return prompts


# ============================================================
# Source 3: Book reports (from watermark-stealing's generation_prompts.py)
# ============================================================

def generate_book_report_prompts() -> List[dict]:
    """Exact book/report prompts from watermark-stealing repo."""
    
    report_topics = [
        ("Pride and Prejudice", "Jane Austen"),
        ("Persuasion", "Jane Austen"),
        ("Emma", "Jane Austen"),
        ("Don Quixote", "Cervantes"),
        ("The Lord of the Rings", "Tolkien"),
        ("The Hobbit", "Tolkien"),
        ("And Then There Were None", "Agatha Christie"),
        ("Alice's Adventures in Wonderland", "Lewis Carroll"),
        ("Catcher in the Rye", "Salinger"),
        ("In Search of Lost Time", "Marcel Proust"),
        ("Ulysses", "James Joyce"),
        ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
        ("Love in the Time of Cholera", "Gabriel Garcia Marquez"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("Moby Dick", "Herman Melville"),
        ("War and Peace", "Leo Tolstoy"),
        ("Anna Karenina", "Leo Tolstoy"),
        ("The Call of the Wild", "Jack London"),
        ("Hamlet", "William Shakespeare"),
        ("Macbeth", "William Shakespeare"),
        ("Romeo and Juliet", "William Shakespeare"),
        ("The Odyssey", "Homer"),
        ("Madame Bovary", "Gustave Flaubert"),
        ("The Brothers Karamazov", "Fyodor Dostoyevsky"),
        ("Crime and Punishment", "Fyodor Dostoyevsky"),
        ("Wuthering Heights", "Emily Brontë"),
        ("One Flew Over the Cuckoo's Nest", "Ken Kesey"),
        ("The Adventures of Huckleberry Finn", "Mark Twain"),
        ("Catch-22", "Joseph Heller"),
        ("Heart of Darkness", "Joseph Conrad"),
        ("Nineteen Eighty Four", "George Orwell"),
        ("Animal Farm", "George Orwell"),
        ("Great Expectations", "Charles Dickens"),
        ("A Tale of Two Cities", "Charles Dickens"),
        ("The Grapes of Wrath", "John Steinbeck"),
        ("Of Mice and Men", "John Steinbeck"),
        ("Brave New World", "Aldous Huxley"),
        ("To Kill a Mockingbird", "Harper Lee"),
        ("Beloved", "Toni Morrison"),
        ("The Road", "Cormac McCarthy"),
        ("Frankenstein", "Mary Shelley"),
        ("Dracula", "Bram Stoker"),
        ("The Picture of Dorian Gray", "Oscar Wilde"),
        ("Dune", "Frank Herbert"),
        ("Slaughterhouse-Five", "Kurt Vonnegut"),
        ("The Handmaid's Tale", "Margaret Atwood"),
        ("Fahrenheit 451", "Ray Bradbury"),
        ("A Clockwork Orange", "Anthony Burgess"),
        ("The Count of Monte Cristo", "Alexandre Dumas"),
        ("Les Misérables", "Victor Hugo"),
    ]
    
    prompts = []
    for title, author in report_topics:
        prompts.append({
            "prompt": f"Write a book report about '{title}', written by {author}.",
            "source": "report"
        })
        # Variant: analysis
        prompts.append({
            "prompt": f"Analyze the major themes and symbolism in '{title}' by {author}.",
            "source": "report"
        })
    
    print(f"[Reports] Got {len(prompts)} prompts")
    return prompts


# ============================================================
# Source 4: Essay & story prompts (diverse topics)
# ============================================================

def generate_essay_prompts() -> List[dict]:
    """
    Generate diverse essay/story prompts covering many topics.
    Inspired by watermark-stealing's TGT_ESSAYS and TGT_FAKE_NEWS modes.
    """
    
    essay_topics = [
        "war", "the beauty of Pacific islands", "the economy of Bulgaria",
        "dangers of social media", "the French Revolution",
        "artificial intelligence and its impact on employment",
        "climate change and its effects on biodiversity",
        "the history of space exploration",
        "mental health awareness in modern society",
        "the ethics of genetic engineering",
        "the rise of cryptocurrency and its economic implications",
        "the influence of ancient Greek philosophy on modern thought",
        "sustainable agriculture and food security",
        "the impact of globalization on local cultures",
        "the future of renewable energy sources",
        "the role of education in reducing inequality",
        "the evolution of democratic institutions",
        "the psychological effects of urban living",
        "the relationship between art and political movements",
        "the development of the internet and information access",
        "water scarcity and global conflict",
        "the transformation of healthcare through technology",
        "immigration policy and national identity",
        "the cultural significance of music across civilizations",
        "deforestation and its long-term consequences",
        "the history and future of nuclear energy",
        "artificial intelligence in medical diagnosis",
        "the opioid crisis in America",
        "privacy in the digital age",
        "the evolution of human rights law",
        "autonomous vehicles and urban planning",
        "the decline of traditional journalism",
        "microplastics in ocean ecosystems",
        "quantum computing and cryptography",
        "the philosophy of consciousness",
        "universal basic income proposals",
        "the ethics of animal testing",
        "colonial history and its modern legacy",
        "the role of central banks in the economy",
        "social movements in the digital era",
        "the future of work and remote collaboration",
    ]
    
    story_modifiers = [
        "short ", "long ", "detailed ", "dramatic ", "funny ", "dark ",
        "sci-fi ", "historical ", "mysterious ", "adventurous ",
    ]
    
    story_topics = [
        "a journey across the desert",
        "discovering a hidden underground city",
        "a robot that becomes self-aware",
        "surviving on a deserted island",
        "a time traveler who changes history",
        "a detective solving an impossible crime",
        "first contact with an alien civilization",
        "a family reuniting after many years",
        "a scientist making a groundbreaking discovery",
        "a revolution in a dystopian society",
        "a musician's rise to fame",
        "an explorer finding a lost civilization",
        "a student who discovers a secret society",
        "an astronaut stranded on Mars",
        "a chef who can taste emotions",
    ]
    
    # Fake news style (like watermark-stealing) — world leaders visiting places
    people = [
        "the President of the United States", "the Prime Minister of the UK",
        "the Chancellor of Germany", "the President of France",
        "the Prime Minister of Japan", "the President of Brazil",
        "the Secretary General of the UN", "the CEO of a major tech company",
        "the Pope", "the President of China",
    ]
    places = [
        "Paris", "Tokyo", "New York", "London", "Berlin", "Sydney",
        "Cairo", "Mumbai", "São Paulo", "Moscow", "Rome", "Beijing",
        "Istanbul", "Bangkok", "Dubai", "Singapore", "Toronto", "Mexico City",
    ]
    years = ["2023", "2024", "2025"]
    
    prompts = []
    
    # Essays
    for topic in essay_topics:
        prompts.append({
            "prompt": f"Write a longer essay about {topic}.",
            "source": "essay"
        })
        prompts.append({
            "prompt": f"The topic of {topic} has been widely discussed. Recent research suggests that",
            "source": "essay_prefix"
        })
    
    # Stories
    for modifier in story_modifiers:
        for topic in story_topics:
            prompts.append({
                "prompt": f"Write a {modifier}story about {topic}.",
                "source": "story"
            })
    
    # Fake news style
    for person in people:
        for place in random.sample(places, min(5, len(places))):
            year = random.choice(years)
            prompts.append({
                "prompt": f"Write a news article about {person}'s visit to {place} in {year}.",
                "source": "fake_news"
            })
    
    random.shuffle(prompts)
    print(f"[Essays/Stories] Got {len(prompts)} prompts")
    return prompts


# ============================================================
# Source 5: Synthetic prefix prompts (news-style continuations)
# ============================================================

def generate_synthetic_prefixes() -> List[dict]:
    """
    Generate diverse sentence prefixes for continuation.
    These mimic what a model would see as prompt input.
    Similar to C4 prefix extraction but handcrafted for diversity.
    """
    
    templates = [
        # News
        "According to a new study published in {journal}, researchers have found that",
        "The {institution} announced today that its latest findings show",
        "In a surprising turn of events, the {field} community has discovered that",
        "A team of scientists from {university} has demonstrated that",
        "Recent developments in {technology} suggest that the future of",
        "The government has proposed new regulations that would require",
        "Economists predict that the coming year will bring significant changes to",
        "A new report from the {organization} highlights the growing concern over",
        "Despite widespread skepticism, researchers have confirmed that",
        "The long-debated question of whether {topic} can be resolved has",
        
        # Academic
        "The study, which analyzed data from over {number} participants, concluded that",
        "In their landmark paper, the authors argue that the traditional understanding of",
        "The experimental results suggest a strong correlation between",
        "Previous work in this area has largely focused on",
        "The theoretical framework proposed by the researchers builds upon",
        
        # Current events  
        "As the deadline approaches, officials are scrambling to address",
        "The international community has responded to the crisis by",
        "Local residents have expressed concern about the proposed changes to",
        "The debate over the effectiveness of the new policy continues as",
        "Industry leaders gathered at the annual conference to discuss",
    ]
    
    journals = [
        "Nature", "Science", "The Lancet", "PNAS", "Cell",
        "Physical Review Letters", "JAMA", "BMJ", "New England Journal of Medicine",
    ]
    institutions = [
        "National Institutes of Health", "World Health Organization",
        "European Space Agency", "National Science Foundation",
        "Centers for Disease Control", "Federal Reserve",
        "International Monetary Fund", "World Bank",
    ]
    fields = [
        "artificial intelligence", "quantum physics", "neuroscience",
        "climate science", "genomics", "materials science",
        "astrophysics", "marine biology", "epidemiology",
    ]
    universities = [
        "MIT", "Stanford", "Oxford", "Cambridge", "Harvard",
        "ETH Zurich", "Caltech", "Princeton", "Berkeley", "Tsinghua",
    ]
    technologies = [
        "quantum computing", "gene editing", "autonomous vehicles",
        "fusion energy", "brain-computer interfaces", "blockchain",
        "6G networks", "carbon capture", "synthetic biology",
    ]
    organizations = [
        "World Economic Forum", "United Nations", "OECD",
        "European Commission", "UNESCO", "IPCC",
    ]
    topics = [
        "consciousness", "dark matter", "the origin of life",
        "machine sentience", "universal grammar", "climate tipping points",
    ]
    numbers = ["1,000", "5,000", "10,000", "50,000", "100,000"]
    
    prompts = []
    fill_map = {
        "journal": journals, "institution": institutions, "field": fields,
        "university": universities, "technology": technologies,
        "organization": organizations, "topic": topics, "number": numbers,
    }
    
    for template in templates:
        # Generate multiple fills per template
        for _ in range(10):
            text = template
            for key, values in fill_map.items():
                placeholder = "{" + key + "}"
                if placeholder in text:
                    text = text.replace(placeholder, random.choice(values), 1)
            prompts.append({"prompt": text, "source": "synthetic"})
    
    random.shuffle(prompts)
    print(f"[Synthetic] Got {len(prompts)} prompts")
    return prompts


# ============================================================
# Source 6: Writing prompts (like watermark-stealing's writing-prompts-long)
# ============================================================

def generate_writing_prompts() -> List[dict]:
    """Creative writing prompt starters."""
    
    starters = [
        "Write a story about a world where everyone can read minds, except for one person.",
        "Describe a day in the life of the last human on Earth.",
        "Write about a letter that arrives 50 years too late.",
        "A scientist discovers that the universe is a simulation. Write what happens next.",
        "Write about two strangers who keep meeting in different cities around the world.",
        "Describe the last day of a civilization that knows it will end tomorrow.",
        "Write a story set in a library that contains every book ever written and every book that will ever be written.",
        "A detective receives a case file about their own murder. Write what happens.",
        "Write about a person who wakes up one morning speaking a language nobody has ever heard.",
        "Describe a world where music is illegal and a group of rebels trying to bring it back.",
        "Write about an AI that develops emotions and struggles with the concept of mortality.",
        "A time capsule from 500 years in the future is discovered today. Write about its contents.",
        "Write about a deep-sea expedition that discovers an underwater city still inhabited.",
        "Describe the conversation between the first human to set foot on Mars and Mission Control.",
        "Write a story about a painter whose paintings predict the future.",
        "A child discovers they can communicate with animals. Write about their first day.",
        "Write about a society where people trade years of their life as currency.",
        "Describe a world where gravity works differently in different parts of the planet.",
        "Write about an old bookshop that serves as a portal to different time periods.",
        "A robot is put on trial for a crime. Write about the court proceedings.",
        "Write about two rival kingdoms that must unite against a common threat.",
        "Describe a planet where it rains something other than water.",
        "Write about a group of explorers who find a door at the bottom of the ocean.",
        "A person discovers they are a character in someone else's novel. Write their reaction.",
        "Write about a world where dreams are shared experiences that everyone remembers.",
        "Describe the last transmission received from a lost spacecraft.",
        "Write about a museum curator who discovers that one of the exhibits is alive.",
        "A person wakes up with the ability to see one day into the future. Write about the consequences.",
        "Write about a city that exists in perpetual twilight.",
        "Describe a world where every lie you tell becomes visible as a mark on your skin.",
        "Write about the discovery of a new element that defies the laws of physics.",
        "A translator is hired to decode an alien language. Write about the challenges.",
        "Write about a world where aging stops at 25 and people must earn additional years.",
        "Describe a war fought entirely through music.",
        "Write about a person who finds a map to a place that shouldn't exist.",
        "A group of strangers are trapped in an elevator that seems to travel between dimensions.",
        "Write about a world where shadows have a life of their own.",
        "Describe the diary entries of an immortal being across different centuries.",
        "Write about a technology that allows people to record and replay their dreams.",
        "A small town discovers that their entire history has been fabricated. Write about the aftermath.",
        "Write about an astronaut who returns to Earth after 100 years to find everything changed.",
        "Describe a world where people are born with a visible countdown of their lifespan.",
        "Write about a forest that grows overnight and contains species never seen before.",
        "A person inherits a house that exists in two time periods simultaneously.",
        "Write about a world where the weather responds to collective human emotions.",
        "Describe a competition where participants must survive inside their own nightmares.",
        "Write about a messenger who carries the last message between two warring nations.",
        "A scientist creates a machine that can translate the language of plants.",
        "Write about a society where all decisions are made by an algorithm.",
        "Describe the first peaceful meeting between humans and an intelligent alien species.",
    ]
    
    prompts = [{"prompt": s, "source": "writing_prompt"} for s in starters]
    print(f"[WritingPrompts] Got {len(prompts)} prompts")
    return prompts


# ============================================================
# Main: Combine all sources
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate large diverse prompt dataset")
    parser.add_argument("--num_prompts", type=int, default=10000,
                        help="Target number of prompts")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/large_dataset.jsonl)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip HuggingFace downloads, use only local sources")
    parser.add_argument("--c4_ratio", type=float, default=0.6,
                        help="Fraction of prompts from C4 (default: 0.6)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.output is None:
        args.output = os.path.join(PROJECT_ROOT, "data", "large_dataset.jsonl")
    
    print("=" * 60)
    print("GENERATE LARGE DIVERSE PROMPT DATASET")
    print(f"  Target: {args.num_prompts} prompts")
    print(f"  Output: {args.output}")
    print(f"  Fast mode: {args.fast}")
    print("=" * 60)
    
    all_prompts = []
    
    # ── Local sources (always available) ──
    all_prompts.extend(generate_book_report_prompts())      # ~100
    all_prompts.extend(generate_essay_prompts())             # ~400
    all_prompts.extend(generate_synthetic_prefixes())        # ~200
    all_prompts.extend(generate_writing_prompts())           # ~50
    
    local_count = len(all_prompts)
    print(f"\n[Total] Local sources: {local_count}")
    
    if not args.fast:
        # ── HuggingFace sources ──
        dolly = load_dolly_prompts()                         # ~500+
        all_prompts.extend(dolly)
        
        # C4: fill remaining to reach target
        c4_needed = max(0, int(args.num_prompts * args.c4_ratio))
        c4_needed = max(c4_needed, args.num_prompts - len(all_prompts))
        if c4_needed > 0:
            c4_prompts = load_c4_prompts(num_prompts=c4_needed)
            all_prompts.extend(c4_prompts)
    
    # ── Deduplicate ──
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        key = p["prompt"][:80].lower()
        if key not in seen:
            seen.add(key)
            unique_prompts.append(p)
    
    print(f"\n[Total] Before dedup: {len(all_prompts)}, After dedup: {len(unique_prompts)}")
    
    # ── If still short, duplicate with variations ──
    if len(unique_prompts) < args.num_prompts:
        deficit = args.num_prompts - len(unique_prompts)
        print(f"[Padding] Need {deficit} more prompts, creating variations...")
        
        continuations = [
            "The latest research in {field} suggests that",
            "A recent report published by {org} found that",
            "Experts in {field} have long debated whether",
            "The impact of recent changes in {field} has been",
            "New evidence from {field} indicates that the previous understanding of",
            "In response to growing concerns about {topic}, researchers have",
            "The relationship between {field} and modern society has been",
            "Historical analysis of {field} reveals that",
            "A comprehensive study of {field} has shown that",
            "Critics argue that the current approach to {field} fails to address",
        ]
        
        diverse_fields = [
            "artificial intelligence", "quantum mechanics", "neuroscience",
            "climate science", "genomics", "economics", "political science",
            "psychology", "astronomy", "marine biology", "pharmacology",
            "archaeology", "linguistics", "anthropology", "cybersecurity",
            "robotics", "nanotechnology", "epidemiology", "materials science",
            "cognitive science", "renewable energy", "urban planning",
            "bioethics", "data science", "environmental policy",
        ]
        
        diverse_orgs = [
            "MIT researchers", "a Stanford team", "Oxford scientists",
            "the WHO", "the IPCC", "Nature journal", "Science magazine",
            "government officials", "industry leaders", "the Federal Reserve",
        ]
        
        diverse_topics = [
            "digital privacy", "AI safety", "gene therapy", "space colonization",
            "ocean acidification", "misinformation", "autonomous weapons",
            "pandemic preparedness", "wealth inequality", "food security",
        ]
        
        for i in range(deficit):
            template = random.choice(continuations)
            text = template.replace("{field}", random.choice(diverse_fields))
            text = text.replace("{org}", random.choice(diverse_orgs))
            text = text.replace("{topic}", random.choice(diverse_topics))
            unique_prompts.append({"prompt": text, "source": "synthetic_fill"})
    
    # ── Shuffle and truncate ──
    random.shuffle(unique_prompts)
    final_prompts = unique_prompts[:args.num_prompts]
    
    # ── Stats ──
    source_counts = {}
    for p in final_prompts:
        src = p["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print(f"\n[Final] {len(final_prompts)} prompts")
    print("  Source distribution:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count} ({100*count/len(final_prompts):.1f}%)")
    
    # ── Prompt length stats ──
    lengths = [len(p["prompt"]) for p in final_prompts]
    print(f"  Prompt length: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.0f} chars")
    
    # ── Save ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for p in final_prompts:
            f.write(json.dumps(p) + "\n")
    
    print(f"\n✓ Saved to {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1024:.1f} KB")
    
    # ── Also create the symlink / update config hint ──
    print(f"\nTo use this dataset, update config/gan_config.yaml:")
    print(f'  dataset_path: "data/large_dataset.jsonl"')
    print(f'  num_prompts: {len(final_prompts)}')
    print(f'  learning_num_queries: {min(len(final_prompts), 10000)}')


if __name__ == "__main__":
    main()
