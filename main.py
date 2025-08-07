from chains.document_qa_chain import build_qa_chain
from agents.contact_agent import conversational_driver
from agents.appointment import book_appointment

def main_chatbot():
    print("Bot: Hi! I can help you with:")
    print("- Document Q&A → type: 'ask q/a'")
    print("- Collecting your contact info → type: 'contact' or 'call me'")
    print("- Booking an appointment → type: 'book'")
    print("- Exit anytime → type: 'exit'\n")

    qa_chain = None

    while True:
        user_input = input("You: ").strip().lower()

        if user_input == "exit":
            print("Bot: Goodbye! Have a great day.")
            break

        elif user_input == "ask q/a":
            if not qa_chain:
                pdf_path = input("Bot: Please enter the path to the document (pdf, txt, docs):\nYou: ").strip()
                try:
                    qa_chain = build_qa_chain(pdf_path)
                    print("Bot: Document loaded! You can now ask questions. Type 'back' to return to the main menu.\n")
                except Exception as e:
                    print(f"Bot: Failed to load document — {e}")
                    continue

            # Q&A loop
            while True:
                question = input("You: ").strip()
                if question.lower() == "back":
                    print("Bot: Returning to main menu...\n")
                    break
                elif question == "":
                    continue
                else:
                    try:
                        response = qa_chain.invoke({"query": question})
                        print("Bot:", response["result"])
                    except Exception as e:
                        print(f"Bot: Something went wrong: {e}")

        elif user_input in ["contact", "call me"]:
            contact_info = conversational_driver()
            if contact_info:
                print("Bot: Thank you! We have stored your contact info.\n")

        elif user_input == "book":
            book_appointment()

        else:
            print("Bot: I didn’t understand that. Please type 'ask q/a', 'contact', 'book', or 'exit'.")

# Run the bot when executing the script directly
if __name__ == "__main__":
    main_chatbot()

