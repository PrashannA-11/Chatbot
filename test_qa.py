from chains.document_qa_chain import build_qa_chain  
if __name__ == "__main__":
    pdf_path = "coffee.pdf"
    
    try:
        qa_chain = build_qa_chain(pdf_path)

        while True:
            query = input("Ask a question (or type 'exit'): ")
            if query.lower() == "exit":
                print("Goodbye!")
                break

            response = qa_chain.invoke({"query": query})

            print("\nAnswer:\n", response["result"])
            print("\nSources:")
            for doc in response["source_documents"]:
                print("-", doc.metadata.get("source", "unknown"))
            print("-" * 50)

    except Exception as e:
        print("Error:", e)
