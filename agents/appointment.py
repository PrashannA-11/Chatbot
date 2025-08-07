import os
import re
import json
from datetime import datetime
from dateparser.search import search_dates

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY

# === LLM SETUP ===
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in config/settings.py")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
output_parser = StrOutputParser()

# === PROMPT CHAINS ===

def generate_confirmation_message(data: dict) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a friendly and natural confirmation message for appointment booking."),
        ("human", "Name: {name}, Phone: {phone}, Email: {email}, Date: {date}")
    ])
    chain = prompt | llm | output_parser
    return chain.invoke(data)

def generate_error_message(field: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful assistant."),
        ("human", "Please generate a polite error message asking the user to re-enter a valid {field}.")
    ])
    chain = prompt | llm | output_parser
    return chain.invoke({"field": field})

# === VALIDATORS ===

def is_valid_email(email: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def is_valid_phone(phone: str) -> bool:
    return re.match(r"^\+?\d{7,15}$", re.sub(r"[ \-\(\)]", "", phone)) is not None

def is_valid_name(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z\s\-']+$", name))

def parse_date(text: str) -> str | None:
    result = search_dates(text, settings={"PREFER_DATES_FROM": "future"})
    if result:
        return result[0][1].strftime("%Y-%m-%d")
    return None

# === DATA COLLECTION ===

def collect_user_data():
    fields = ["name", "phone", "email", "date"]
    prompts = {
        "name": "What is your full name?",
        "phone": "What is your phone number?",
        "email": "What is your email address?",
        "date": "What date would you prefer for the appointment?"
    }

    validators = {
        "name": is_valid_name,
        "phone": is_valid_phone,
        "email": is_valid_email,
        "date": lambda x: True  
    }

    user_data = {}

    for field in fields:
        while True:
            print(f"Bot: {prompts[field]}")
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                return None

            if not user_input:
                print("Bot: Input cannot be empty.")
                continue

            if field == "date":
                parsed_date = parse_date(user_input)
                if parsed_date:
                    user_data[field] = parsed_date
                    break
                else:
                    print("Bot: I couldn't understand that date. Please try again.")
                    continue

            elif validators[field](user_input):
                user_data[field] = user_input
                break
            else:
                print("Bot:", generate_error_message(field))

    return user_data

# === CONFIRMATION STEP ===

def confirm_appointment(data: dict) -> bool:
    print("\nBot:", generate_confirmation_message(data))
    print("Bot: Is this information correct? (yes/no)")

    while True:
        response = input("You: ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n", "exit"]:
            return False
        else:
            print("Bot: Please respond with 'yes' or 'no'.")

# === SAVE DATA TO JSON ===

def save_appointment(data: dict, filename="appointments.json"):
    # Read existing data
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                appointments = json.load(f)
                if not isinstance(appointments, list):
                    appointments = []
            except json.JSONDecodeError:
                appointments = []
    else:
        appointments = []

    # Append new appointment data
    appointments.append(data)

    # Write back to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(appointments, f, indent=4)

# === MAIN FLOW ===

def book_appointment():
    print("Bot: Hi there! I can help you schedule an appointment.")
    print("Type 'exit' anytime to stop.\n")

    while True:
        data = collect_user_data()
        if data is None:
            print("Bot: Exiting. Thank you!")
            break

        if confirm_appointment(data):
            save_appointment(data)
            print("Bot: Great! Your appointment has been booked. We'll reach out to you soon.")
            break
        else:
            print("Bot: No problem. Let's try again.\n")

# === ENTRY POINT ===

if __name__ == "__main__":
    book_appointment()
