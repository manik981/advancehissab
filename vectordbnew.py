# vectordb.py - Advanced Vector Database Engine with Dual DB Support
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import random
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_FILE_PATH = "hissab_vector_db.pkl"
BAD_DB_FILE_PATH = "bad_prompts_db.pkl" # Galtiyon ke liye naya DB
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Global Variables ---
embedding_model = SentenceTransformer(MODEL_NAME)
hissab_db = None
bad_prompts_db = None

# --- Initial Data Prompts (Enhanced for Hindi understanding) ---
INITIAL_PROMPTS = [
{
        "category": "personal_expense_tracking",
        "user_text": "Aaj 30 rupaye bus ke lage, 250 ka khana khaya, aur 500 ki ek shirt kharidi.",
        "model_response": "Aapke aaj ke kul kharch is prakaar hain:\n- Bus: ₹30\n- Khana: ₹250\n- Shirt: ₹500\n**Kul Kharch: ₹830**"
    },
    {
        "category": "group_settlement",
        "user_text": "Hum 3 dost, main, Rohit aur Suman, Goa gaye. Maine hotel ke 6000 diye, Rohit ne khaane ke 3000 kharch kiye. Suman ne kuch nahi diya.",
        "model_response": "**Trip ka Hisaab:**\n- **Kul Kharch:** ₹9,000 (₹6000 Hotel + ₹3000 Khaana)\n- **Log:** 3\n- **Prati Vyakti Hissa:** ₹3,000\n**Settlement:**\n- Aapne ₹3,000 extra diye hain.\n- Rohit ne apna hissa de diya hai.\n- **Isliye, Suman ko aapko ₹3,000 dene hain.**"
    },
    {
        "category": "monthly_budget_and_savings",
        "user_text": "Meri salary 50000 hai. 15000 ghar ka kiraya, 3000 bijli paani, 10000 ration aur 5000 aane jaane mein lag jaata hai. Kitni bachat hoti hai?",
        "model_response": "**Aapka Maheene ka Hisaab:**\n- **Kul Aamdani (Salary):** ₹50,000\n- **Kul Kharch:** ₹33,000 (₹15000 Kiraya + ₹3000 Bill + ₹10000 Ration + ₹5000 Aana-Jaana)\n- **Isliye, aapki kul bachat ₹17,000 hai.**"
    },
    {
        "category": "price_comparison",
        "user_text": "Ek dukaan par Samsung ka phone 25000 ka hai aur wahi phone doosri dukaan par 23500 ka mil raha hai. Kaunsa sasta hai?",
        "model_response": "**Cheezon ki Tulna:**\n- Dusri dukaan wala phone sasta hai.\n- **Dono ke beech ₹1,500 ka antar hai.**"
    },
    {
        "category": "lending_and_borrowing",
        "user_text": "Maine apne dost Aman ko 2000 rupaye udhaar diye the, usne mujhe 500 lauta diye hain. Ab kitne paise lene baaki hain?",
        "model_response": "**Udhaari ka Hisaab:**\n- **Kul Udhaar:** ₹2,000\n- **Vaapas Mile:** ₹500\n- **Isliye, aapko Aman se abhi ₹1,500 aur lene hain.**"
    },
    {
        "category": "investment_and_profit",
        "user_text": "Maine 50,000 rupaye share market mein lagaye the, aur ab unki value 65,000 ho gayi hai. Mujhe kitna munafa hua?",
        "model_response": "**Nivesh ka Hisaab:**\n- **Aapka Nivesh:** ₹50,000\n- **Abhi ka Value:** ₹65,000\n- **Isliye, aapko kul ₹15,000 ka munafa hua hai.**"
    },
    {
        "category": "loan_and_emi",
        "user_text": "Mera 2 lakh ka personal loan hai aur har mahine 5000 ki EMI jaati hai. 1 saal mein main kitna chuka dunga?",
        "model_response": "**Loan ka Hisaab:**\n- **Har Mahine ki EMI:** ₹5,000\n- **Ek Saal (12 Mahine) mein Kul Bhugtaan:** ₹60,000 (5000 x 12)\n- **Isliye, aap ek saal mein ₹60,000 chuka denge.**"
    },
    {
        "category": "income_and_balance",
        "user_text": "Mere account mein 10000 the, mujhe 25000 salary mili. Uske baad maine 2000 ka bill bhara. Ab kitne bache?",
        "model_response": "**Account ka Hisaab:**\n- **Shuruaati Balance:** ₹10,000\n- **Salary Aayi:** + ₹25,000\n- **Bill Bhara:** - ₹2,000\n- **Aapka abhi ka balance ₹33,000 hai.**"
    },
    {
        "category": "discount_and_offers",
        "user_text": "Ek jacket 4000 ki hai aur us par 20% ka discount hai. Mujhe kitne paise dene honge?",
        "model_response": "**Discount ka Hisaab:**\n- **Jacket ka Daam:** ₹4,000\n- **Discount (20%):** ₹800 (4000 ka 20%)\n- **Isliye, aapko ₹3,200 dene honge.**"
    },
    {
        "category": "salary_calculation",
        "user_text": "Main din ke 800 rupaye kamata hoon. Is mahine maine 25 din kaam kiya. Meri is mahine ki salary kitni hui?",
        "model_response": "**Salary ka Hisaab:**\n- **Ek Din ki Kamai:** ₹800\n- **Kul Kaam ke Din:** 25\n- **Isliye, aapki is mahine ki salary ₹20,000 hui (800 x 25).**"
    }
]

# --- Database Setup (Good & Bad) ---

def _initialize_database():
    """Initializes the main 'good prompts' database."""
    print("Naya 'Good Prompts' Vector DB banaya ja raha hai...")
    df = pd.DataFrame(INITIAL_PROMPTS)
    embeddings = embedding_model.encode(df['user_text'].tolist())
    df['embedding'] = list(embeddings)
    df.to_pickle(DB_FILE_PATH)
    print(f"Vector DB '{DB_FILE_PATH}' mein save ho gaya hai.")
    return df

def setup_vector_db():
    """Loads or initializes the main 'good prompts' vector database."""
    global hissab_db
    if os.path.exists(DB_FILE_PATH):
        hissab_db = pd.read_pickle(DB_FILE_PATH)
    else:
        hissab_db = _initialize_database()

def setup_bad_prompts_db():
    """Loads or initializes the separate 'bad prompts' database."""
    global bad_prompts_db
    if os.path.exists(BAD_DB_FILE_PATH):
        bad_prompts_db = pd.read_pickle(BAD_DB_FILE_PATH)
    else:
        print("Naya 'Bad Prompts' DB banaya ja raha hai...")
        # Ismein model ka response bhi store hoga
        df = pd.DataFrame(columns=['user_text', 'model_response', 'embedding'])
        df.to_pickle(BAD_DB_FILE_PATH)
        bad_prompts_db = df

# --- Core Logic Functions ---

def get_category_from_prompt(user_prompt: str) -> str:
    """Uses an LLM to classify the user's prompt into one of the predefined categories."""
    global hissab_db
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: raise ValueError("API Key missing.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Chhota model classification ke liye theek hai
    
    category_list = hissab_db['category'].unique().tolist()
    
    classification_prompt = f"""
    You are a financial query classification expert. Classify the user query into one of the following categories. Return only the category name.
    Available Categories: {', '.join(category_list)}
    User Query: "{user_prompt}"
    Category:
    """
    
    try:
        response = model.generate_content(classification_prompt)
        category = response.text.strip().lower().replace(" ", "_")
        return category if category in category_list else "personal_expense_tracking"
    except Exception as e:
        print(f"Category classification error: {e}")
        return "personal_expense_tracking" # Fallback

def find_random_examples_from_category(category: str, max_examples: int = 5) -> list:
    """
    Finds a given category in the DB and returns a random sample of examples from it.
    """
    global hissab_db
    if hissab_db is None or hissab_db.empty: return []

    category_df = hissab_db[hissab_db['category'] == category]
    if category_df.empty: return []

    # Jitne examples hain, ya max_examples, jo bhi kam ho, utne random samples nikalo
    num_samples = min(max_examples, len(category_df))
    random_samples = category_df.sample(n=num_samples)
    
    return random_samples[['user_text', 'model_response']].to_dict(orient='records')

# --- Functions for Adding Data to Databases ---

def add_user_prompt_to_db(hinglish_user_prompt: str):
    """
    Adds a new 'good' user prompt to the main vector DB under the best-matching category.
    """
    global hissab_db
    print(f"Naya prompt 'Good DB' mein add kiya ja raha hai: '{hinglish_user_prompt}'")
    
    # Pehle LLM se iski category pata karo
    category = get_category_from_prompt(hinglish_user_prompt)
    embedding = embedding_model.encode([hinglish_user_prompt])[0]
    
    # Nayi entry
    new_entry = pd.DataFrame([{
        'category': category, 
        'user_text': hinglish_user_prompt, 
        'embedding': embedding, 
        'model_response': '' # User-added prompts ka response nahi hota
    }])
    
    hissab_db = pd.concat([hissab_db, new_entry], ignore_index=True)
    hissab_db.to_pickle(DB_FILE_PATH)
    print(f"'{category}' category mein prompt save ho gaya.")

def add_to_bad_prompts_db(hinglish_user_prompt: str, model_response: str):
    """Adds a prompt-response pair marked as 'bad' by the user to the bad_prompts_db."""
    global bad_prompts_db
    print(f"Galtiyon wala prompt 'Bad DB' mein add kiya ja raha hai: '{hinglish_user_prompt}'")
    
    embedding = embedding_model.encode([hinglish_user_prompt])[0]
    
    # Nayi entry (ismein model_response bhi hai)
    new_entry = pd.DataFrame([{
        'user_text': hinglish_user_prompt,
        'model_response': model_response,
        'embedding': embedding
    }])

    bad_prompts_db = pd.concat([bad_prompts_db, new_entry], ignore_index=True)
    bad_prompts_db.to_pickle(BAD_DB_FILE_PATH)
    print("'Bad DB' update ho gaya hai.")

# --- Helper Function ---
def is_bad_prompts_db_empty():
    """Checks if the bad prompts database has any entries."""
    global bad_prompts_db
    return bad_prompts_db is None or bad_prompts_db.empty
