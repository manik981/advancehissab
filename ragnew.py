# rag.py - Optimized Retrieval-Augmented Generation Engine
from vectordb import find_random_examples_from_category

def get_enhanced_prompt(hinglish_user_story: str, category: str) -> str:
    """
    Creates a RAG-enhanced prompt using a pre-determined category and random example sampling.
    This version is faster as it does not perform classification.
    """
    print(f"✅ Using pre-identified Category from main: '{category}'")
    
    # Step 1: Di gayi category se maximum 5 random examples retrieve karein.
    random_examples = find_random_examples_from_category(category, max_examples=5)
    print(f"✅ Retrieved {len(random_examples)} random examples from '{category}'.")

    # Step 2: LLM ke liye final, tuned prompt taiyaar karein.
    final_prompt = """You are 'HissabGPT', an AI expert specializing in Indian personal and group finance calculations from Hinglish text.

**Primary Directive:** Your ONLY task is to act as a calculator. Analyze the user's story, identify all financial transactions, and provide a clear, step-by-step summary in simple Hindi.

**Critical Instruction on Voice Errors:** User text often comes from voice-to-text and has errors. You MUST interpret words based on financial context.
- If you see "bachche" (children), you MUST assume the user meant "bache" (remaining balance).
- If a word seems out of place, think of a financially relevant word that sounds similar.
- NEVER comment on the user's language or potential errors. Just provide the correct calculation as if the text was perfect.

**Output Format:**
1. Start with a clear title (e.g., "**Aapka Hisaab:**").
2. List all transactions with amounts.
3. Show the final calculation clearly.
4. Bold the final result.

"""

    if random_examples:
        final_prompt += "Follow the format from these examples precisely:\n\n"
        for i, example in enumerate(random_examples, 1):
            final_prompt += f"--- EXAMPLE {i} ---\n"
            final_prompt += f"User Text: \"{example['user_text']}\"\n"
            final_prompt += f"Your Response:\n{example['model_response']}\n"
            final_prompt += f"--- END EXAMPLE {i} ---\n\n"

    final_prompt += "--- USER'S FINAL TASK ---\n"
    final_prompt += f"User Text: \"{hinglish_user_story}\"\n"
    final_prompt += "Your Response:\n"
    
    return final_prompt

