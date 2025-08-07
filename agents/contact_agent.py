import os
import re
import json
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY

# Setup
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in config/settings.py")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# LLM initialization
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Validators
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    normalized = re.sub(r"[ \-\(\)]", "", phone)
    return re.match(r"^\+?\d{7,15}$", normalized)

def is_valid_name(name):
    return re.match(r"^[A-Za-z\s]+$", name)

# Prompt for LLM (only for error messages)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. The user provided invalid input for a contact form field.
Respond with a brief, polite message asking them to provide valid input. Be encouraging and friendly.
Do not ask any other questions or provide examples."""),
    ("human", "The user gave invalid {field_type}: '{user_input}'. Please ask them to re-enter a valid {field_type}.")
])

# LangChain chain
chain = prompt | llm | StrOutputParser()

# Save to JSON file
def save_contact_info(user_data, filename="contacts.json"):
    path = Path(filename)
    contacts = []

    if path.exists():
        with open(filename, "r", encoding="utf-8") as f:
            try:
                contacts = json.load(f)
            except json.JSONDecodeError:
                contacts = []

    contacts.append(user_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(contacts, f, indent=2)

# Conversation driver
def conversational_driver():
    user_data = {"name": "", "phone": "", "email": ""}
    fields = ["name", "phone", "email"]
    prompts = {
        "name": "Can you please tell me your full name?",
        "phone": "Thanks! Now, your phone number please.",
        "email": "Great! Finally, could you give your email address?"
    }
    validators = {
        "name": is_valid_name,
        "phone": is_valid_phone,
        "email": is_valid_email
    }

    print("Bot: Hi! I'll help collect your contact info. Type 'exit' anytime to quit.\n")
    current_index = 0

    while current_index < len(fields):
        current_field = fields[current_index]
        print("Bot:", prompts[current_field])
        user_reply = input("You: ").strip()

        if user_reply.lower() == "exit":
            print("Bot: Exiting. Thank you!")
            return

        validator = validators[current_field]
        if validator(user_reply):
            user_data[current_field] = user_reply
            current_index += 1
        else:
            # Use LLM to respond to invalid input politely
            try:
                clarification = chain.invoke({
                    "field_type": current_field,
                    "user_input": user_reply
                }).strip()
                print("Bot:", clarification)
            except Exception:
                # Fallback message if LLM fails
                print(f"Bot: Please provide a valid {current_field}.")

    # All fields collected — show confirmation (NO LLM here)
    print("\nBot: Here's what I have:")
    print(f"- Name: {user_data['name']}")
    print(f"- Phone: {user_data['phone']}")
    print(f"- Email: {user_data['email']}")
    print("Is this correct? (yes/no)")

    reply = input("You: ").strip().lower()
    if reply == "yes":
        save_contact_info(user_data)
        print("Bot: Thank you! We will contact you soon.")
    else:
        print("Bot: Okay, let's start over.\n")
        conversational_driver()

# Entry point
if __name__ == "__main__":
    conversational_driver()
