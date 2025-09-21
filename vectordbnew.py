# vectordbnew.py - Upgraded Vector DB Engine with Hybrid Retrieval & Structured Logging
import os
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_FILE_PATH = "hissab_vector_db.pkl"
BAD_DB_FILE_PATH = "bad_prompts_db.pkl"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Global Variables ---
embedding_model = SentenceTransformer(MODEL_NAME)
hissab_db = None
bad_prompts_db = None
category_embeddings = None

# --- Initial Data & Category Descriptions ---
INITIAL_PROMPTS = [
    {"category": "personal_expense_tracking", "user_text": "Aaj 30 rupaye bus ke lage, 250 ka khana khaya, aur 500 ki ek shirt kharidi.", "model_response": "Aapke aaj ke kul kharch is prakaar hain:\n- Bus: ₹30\n- Khana: ₹250\n- Shirt: ₹500\n**Kul Kharch: ₹830**"},
    {"category": "group_settlement", "user_text": "Hum 3 dost, main, Rohit aur Suman, Goa gaye. Maine hotel ke 6000 diye, Rohit ne khaane ke 3000 kharch kiye.", "model_response": "**Trip ka Hisaab:**\n- **Kul Kharch:** ₹9,000\n- **Log:** 3\n- **Prati Vyakti Hissa:** ₹3,000\n**Settlement:**\n- Aapne ₹3,000 extra diye hain.\n- Rohit ne apna hissa de diya hai.\n- **Isliye, Suman ko aapko ₹3,000 dene hain.**"},
    {"category": "monthly_budget_and_savings", "user_text": "Meri salary 50000 hai. 15000 ghar ka kiraya, 3000 bijli paani, 10000 ration aur 5000 aane jaane mein lag jaata hai. Kitni bachat hoti hai?", "model_response": "**Aapka Maheene ka Hisaab:**\n- **Kul Aamdani (Salary):** ₹50,000\n- **Kul Kharch:** ₹33,000 (₹15000 Kiraya + ₹3000 Bill + ₹10000 Ration + ₹5000 Aana-Jaana)\n- **Isliye, aapki kul bachat ₹17,000 hai.**"},
    {"category": "price_comparison", "user_text": "Ek dukaan par Samsung ka phone 25000 ka hai aur wahi phone doosri dukaan par 23500 ka mil raha hai. Kaunsa sasta hai?", "model_response": "**Cheezon ki Tulna:**\n- Dusri dukaan wala phone sasta hai.\n- **Dono ke beech ₹1,500 ka antar hai.**"},
    {"category": "lending_and_borrowing", "user_text": "Maine apne dost Aman ko 2000 rupaye udhaar diye the, usne mujhe 500 lauta diye hain. Ab kitne paise lene baaki hain?", "model_response": "**Udhaari ka Hisaab:**\n- **Kul Udhaar:** ₹2,000\n- **Vaapas Mile:** ₹500\n- **Isliye, aapko Aman se abhi ₹1,500 aur lene hain.**"},
    {"category": "investment_and_profit", "user_text": "Maine 50,000 rupaye share market mein lagaye the, aur ab unki value 65,000 ho gayi hai. Mujhe kitna munafa hua?", "model_response": "**Nivesh ka Hisaab:**\n- **Aapka Nivesh:** ₹50,000\n- **Abhi ka Value:** ₹65,000\n- **Isliye, aapko kul ₹15,000 ka munafa hua hai.**"},
    {"category": "loan_and_emi", "user_text": "Mera 2 lakh ka personal loan hai aur har mahine 5000 ki EMI jaati hai. 1 saal mein main kitna chuka dunga?", "model_response": "**Loan ka Hisaab:**\n- **Har Mahine ki EMI:** ₹5,000\n- **Ek Saal (12 Mahine) mein Kul Bhugtaan:** ₹60,000 (5000 x 12)\n- **Isliye, aap ek saal mein ₹60,000 chuka denge.**"},
    {"category": "income_and_balance", "user_text": "Mere account mein 10000 the, mujhe 25000 salary mili. Uske baad maine 2000 ka bill bhara. Ab kitne bache?", "model_response": "**Account ka Hisaab:**\n- **Shuruaati Balance:** ₹10,000\n- **Salary Aayi:** + ₹25,000\n- **Bill Bhara:** - ₹2,000\n- **Aapka abhi ka balance ₹33,000 hai.**"},
    {"category": "discount_and_offers", "user_text": "Ek jacket 4000 ki hai aur us par 20% ka discount hai. Mujhe kitne paise dene honge?", "model_response": "**Discount ka Hisaab:**\n- **Jacket ka Daam:** ₹4,000\n- **Discount (20%):** ₹800 (4000 ka 20%)\n- **Isliye, aapko ₹3,200 dene honge.**"},
    {"category": "salary_calculation", "user_text": "Main din ke 800 rupaye kamata hoon. Is mahine maine 25 din kaam kiya. Meri is mahine ki salary kitni hui?", "model_response": "**Salary ka Hisaab:**\n- **Ek Din ki Kamai:** ₹800\n- **Kul Kaam ke Din:** 25\n- **Isliye, aapki is mahine ki salary ₹20,000 hui (800 x 25).**"},
    {"category": "income_and_balance", "user_text": "Mere paas 1000 rupay the, maine 300 kharch kar diye, ab kitne bachche hain?", "model_response": "**Account ka Hisaab:**\n- **Shuruaati Balance:** ₹1,000\n- **Kharch:** - ₹300\n- **Aapke paas ab ₹700 bache hain.**"}
]

CATEGORY_DESCRIPTIONS = {
    "personal_expense_tracking": "Calculating total daily expenses for one person.",
    "group_settlement": "Settling shared expenses between multiple friends on a trip or event.",
    "monthly_budget_and_savings": "Calculating monthly savings based on salary and recurring bills like rent.",
    "price_comparison": "Comparing the prices of two or more items to find which is cheaper.",
    "lending_and_borrowing": "Tracking money given or taken as a loan between individuals.",
    "investment_and_profit": "Calculating profit or loss from investments like stocks.",
    "loan_and_emi": "Calculating loan repayments and Equated Monthly Installments (EMI).",
    "income_and_balance": "Calculating the final balance after income and expenses in an account.",
    "discount_and_offers": "Calculating the final price after applying a percentage discount.",
    "salary_calculation": "Calculating total salary based on daily or hourly wages.",
    "unknown": "A general financial query that does not fit other categories."
}

def _initialize_database():
    global category_embeddings
    print("Naya 'Good Prompts' Vector DB banaya ja raha hai...")
    df = pd.DataFrame(INITIAL_PROMPTS)
    df['embedding'] = list(embedding_model.encode(df['user_text'].tolist()))
    df.to_pickle(DB_FILE_PATH)
    
    categories = list(CATEGORY_DESCRIPTIONS.keys())
    descriptions = list(CATEGORY_DESCRIPTIONS.values())
    cat_embeds = embedding_model.encode(descriptions)
    category_embeddings = {cat: emb for cat, emb in zip(categories, cat_embeds)}
    return df

def setup_vector_db():
    global hissab_db, category_embeddings
    if os.path.exists(DB_FILE_PATH):
        hissab_db = pd.read_pickle(DB_FILE_PATH)
        if category_embeddings is None:
            categories = list(CATEGORY_DESCRIPTIONS.keys())
            descriptions = list(CATEGORY_DESCRIPTIONS.values())
            cat_embeds = embedding_model.encode(descriptions)
            category_embeddings = {cat: emb for cat, emb in zip(categories, cat_embeds)}
    else:
        hissab_db = _initialize_database()

def setup_bad_prompts_db():
    global bad_prompts_db
    if os.path.exists(BAD_DB_FILE_PATH):
        bad_prompts_db = pd.read_pickle(BAD_DB_FILE_PATH)
    else:
        print("Naya 'Bad Prompts' DB banaya ja raha hai...")
        df = pd.DataFrame(columns=['log_data'])
        df.to_pickle(BAD_DB_FILE_PATH)
        bad_prompts_db = df

def find_semantic_categories(user_prompt: str, top_k: int = 2) -> list:
    global category_embeddings
    if not category_embeddings: return []
    user_embedding = embedding_model.encode([user_prompt])[0]
    categories = list(category_embeddings.keys())
    embeddings = np.array(list(category_embeddings.values()))
    similarities = cosine_similarity([user_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [categories[i] for i in top_indices]

def find_random_examples_from_category(category: str, max_examples: int = 5, min_examples: int = 1) -> list:
    global hissab_db
    category_df = hissab_db[hissab_db['category'] == category]
    if category_df.empty: return []
    num_samples = min(max_examples, len(category_df))
    if num_samples < min_examples: return []
    return category_df.sample(n=num_samples)[['user_text', 'model_response']].to_dict(orient='records')

def add_user_prompt_to_db(hinglish_prompt: str, model_response: str, primary_category: str):
    global hissab_db
    print(f"Naya example '{primary_category}' category mein add kiya ja raha hai...")
    embedding = embedding_model.encode([hinglish_prompt])[0]
    new_example = pd.DataFrame([{'category': primary_category, 'user_text': hinglish_prompt, 'model_response': model_response, 'embedding': embedding}])
    hissab_db = pd.concat([hissab_db, new_example], ignore_index=True)
    hissab_db.to_pickle(DB_FILE_PATH)
    print("Naya example 'Good DB' mein save ho gaya.")

def add_to_bad_prompts_db(log_data: dict):
    global bad_prompts_db
    print("Galti ka structured log 'Bad DB' mein save kiya ja raha hai...")
    new_log = pd.DataFrame([{'log_data': log_data}])
    bad_prompts_db = pd.concat([bad_prompts_db, new_log], ignore_index=True)
    bad_prompts_db.to_pickle(BAD_DB_FILE_PATH)
    print("'Bad DB' update ho gaya hai.")

def get_all_categories() -> list:
    return list(CATEGORY_DESCRIPTIONS.keys())

# --- NAYA CHANGE: YEH FUNCTION ADD KIYA GAYA HAI ---
def is_bad_prompts_db_empty():
    """Checks if the bad prompts database has any entries."""
    global bad_prompts_db
    return bad_prompts_db is None or bad_prompts_db.empty

