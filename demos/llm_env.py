import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
from bs4 import BeautifulSoup

# Initialize LLM
print('Model loading')
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
print('Model loaded')

# Database setup
conn = sqlite3.connect("knowledge.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS pages (id INTEGER PRIMARY KEY, url TEXT, content TEXT, keywords TEXT, timestamp DATETIME)")
print('DB connected')

def scrape_page(url: str):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(strip=True, separator="\n")
        return text[:3000]  # Truncate to fit LLM context
    except:
        return None


def web_search(query: str):
    # Use Brave/Google API or scrape SERP
    res = requests.get(f'https://duckduckgo.com/html/?q={query}')

    soup = BeautifulSoup(res.text, "html.parser")
    return [{"url": r.find('a').get('href'), "snippet": r.find('h2').text} for r in soup.find_all('div', class_='result')]


def summarize(text: str, instruction: str):
    messages = [{'role': 'user', "content": f"{instruction}\n{text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Orchestrator loop
while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    
    # Generate command via LLM
    prompt = f"""Respond with a command (SEARCH/SAVE/RECALL).
    {user_input}"""
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=50)
    command = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f'\033[31m{command}\033[0m')
    command = command.split("LLM: ")[-1]
    # Execute command
    if command.startswith("SEARCH:"):
        query = command.split("SEARCH: ")[1]
        results = web_search(query)
        for result in results:
            content = scrape_page(result["url"])
            if content:
                summary = summarize(content, "Summarize this briefly:")
                cursor.execute("INSERT INTO pages (url, content, keywords, timestamp) VALUES (?, ?, ?, datetime('now'))",
                               (result["url"], summary, query))
        conn.commit()
    elif command.startswith("RECALL:"):
        keyword = command.split("RECALL: ")[1]
        cursor.execute("SELECT content FROM pages WHERE keywords LIKE ?", (f"%{keyword}%",))
        print("Retrieved:", cursor.fetchall())
