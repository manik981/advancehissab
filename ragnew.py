# rag.py - Advanced Retrieval-Augmented Generation Engine

# This module now uses a multi-step process to create a highly contextual prompt:
# 1. Classify the user's query into a predefined category using an LLM.
# 2. Retrieve a random sample of examples from that specific category.
# 3. Construct a detailed "few-shot" prompt with these random examples to guide the main LLM.

from vectordb import get_category_from_prompt, find_random_examples_from_category

def get_enhanced_prompt(hinglish_user_story: str) -> str:
    """
    Creates a RAG-enhanced prompt using category classification and random example sampling.
    """
    # Step 1: LLM ka istemal karke user ke prompt ki category pata karein.
    # Yeh kaam ab 'vectordb.py' mein hota hai.
    category = get_category_from_prompt(hinglish_user_story)
    print(f"✅ Identified Category: '{category}'")

    # Step 2: Chuni gayi category se maximum 5 random examples nikalein.
    # Yeh semantic search se alag hai, aur database ke badhne par bhi prompt ko chhota rakhega.
    random_examples = find_random_examples_from_category(category, max_examples=5)
    print(f"✅ Retrieved {len(random_examples)} random examples from '{category}'.")

    # Step 3: LLM ke liye final prompt taiyaar karein.
    
    # Base instruction, jismein voice-to-text ki galtiyon ke liye warning bhi shaamil hai.
    final_prompt = """You are an expert financial assistant. Your primary task is to analyze a user's story in Hinglish and provide a clear, step-by-step financial summary in Hindi. Please use the user's currency and values accurately.
**Important Note:** The user's text may come from a voice-to-text system and can contain phonetic errors (e.g., transcribing 'बचे' as 'बच्चे'). Please prioritize the financial context to understand the user's true intent.

"""

    # Agar random examples mile hain, to unhe prompt mein jodein.
    if random_examples:
        final_prompt += "Use the following examples to understand the required format and calculation style.\n\n"
        
        for i, example in enumerate(random_examples, 1):
            final_prompt += f"--- EXAMPLE {i} ---\n"
            final_prompt += f"User Text: \"{example['user_text']}\"\n"
            final_prompt += f"Your Response:\n{example['model_response']}\n"
            final_prompt += f"--- END EXAMPLE {i} ---\n\n"

    # Examples ke baad, user ka actual sawaal jodein.
    final_prompt += "Now, analyze the following user's story and provide the financial summary in the same way.\n\n"
    final_prompt += "--- FINAL TASK ---\n"
    final_prompt += f"User Text: \"{hinglish_user_story}\"\n"
    final_prompt += "Your Response:\n"

    return final_prompt