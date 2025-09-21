# ragnew.py

from typing import List, Dict

MAX_EXAMPLES_PER_CATEGORY = 5  # Optional: cap examples per category

def sanitize_text(text: str) -> str:
    """Sanitize risky characters like triple quotes."""
    return text.replace('"""', '\"\"\"')

def format_examples(examples_by_category: Dict[str, List[Dict[str, str]]]) -> List[str]:
    """Format examples grouped by category with continuous numbering."""
    lines = []
    example_counter = 1

    for category, examples in examples_by_category.items():
        lines.append(f"\n--- EXAMPLES FROM CATEGORY: {category} ---")
        for example in examples[:MAX_EXAMPLES_PER_CATEGORY]:
            lines.append(f"--- EXAMPLE {example_counter} ---")
            lines.append(f'User Text: "{example["user_text"]}"')
            lines.append("Your Response:")
            lines.append(example["model_response"])
            lines.append("")  # spacing
            example_counter += 1

    return lines

def get_enhanced_prompt(
    hinglish_user_story: str,
    primary_category: str,
    semantic_categories: List[str],
    examples_by_category: Dict[str, List[Dict[str, str]]]
) -> str:
    """
    Formats all retrieved information into a final prompt for HissabGPT.
    """
    hinglish_user_story = sanitize_text(hinglish_user_story)

    prompt_lines = [
        "You are 'HissabGPT', an AI expert specializing in Indian personal and group finance calculations.",
        "",
        "**User's Query Analysis:**",
        f'- User\'s Query (Hinglish): "{hinglish_user_story}"',
        f"- Primary Identified Category: {primary_category}",
        f"- Top 2 Semantically Similar Categories: {', '.join(semantic_categories)}",
        "",
        "**Critical Instructions:**",
        "1.  Your ONLY task is to act as a calculator based on the user's query.",
        "2.  The text may have voice-to-text errors (e.g., 'bachche' for 'bache'). You MUST interpret based on financial context and NEVER comment on the errors.",
        "3.  Use the provided examples from the semantically similar categories to understand the required Hindi output format.",
        "4.  Provide a clear, step-by-step summary. Bold the final result.",
        "",
        "**Reference Examples:**"
    ]

    if examples_by_category:
        prompt_lines.extend(format_examples(examples_by_category))
    else:
        prompt_lines.append("No relevant examples found. Analyze the query based on general knowledge.")

    prompt_lines.extend([
        "--- USER'S FINAL TASK ---",
        "Analyze the User's Query and provide the financial summary in simple Hindi.",
        "Your Response:"
    ])

    final_prompt = "\n".join(prompt_lines)
    return final_prompt
