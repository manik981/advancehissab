# ragnew.py - Prompt Formatting Engine

def get_enhanced_prompt(
    hinglish_user_story: str, 
    primary_category: str,
    semantic_categories: list,
    examples: list
) -> str:
    """
    Formats all the retrieved information into a final, detailed prompt for the LLM.
    """
    final_prompt = f"""You are 'HissabGPT', an AI expert specializing in Indian personal and group finance calculations.

**User's Query Analysis:**
- User's Query (Hinglish): "{hinglish_user_story}"
- Primary Identified Category: {primary_category}
- Top 2 Semantically Similar Categories: {', '.join(semantic_categories)}

**Critical Instructions:**
1.  Your ONLY task is to act as a calculator based on the user's query.
2.  The text may have voice-to-text errors (e.g., 'bachche' for 'bache'). You MUST interpret based on financial context and NEVER comment on the errors.
3.  Use the provided examples to understand the required Hindi output format.
4.  Provide a clear, step-by-step summary. Bold the final result.

**Reference Examples:**
"""

    if examples:
        for i, example in enumerate(examples, 1):
            final_prompt += f"--- EXAMPLE {i} ---\n"
            final_prompt += f"User Text: \"{example['user_text']}\"\n"
            final_prompt += f"Your Response:\n{example['model_response']}\n"
    else:
        final_prompt += "No relevant examples found. Analyze the query based on general knowledge.\n"

    final_prompt += f"""
--- USER'S FINAL TASK ---
Analyze the User's Query and provide the financial summary in simple Hindi.
Your Response:
"""
  
  return final_prompt
