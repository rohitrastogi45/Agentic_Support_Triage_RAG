from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from triage_and_retrieval import TriageAgent 

class ResponseGenerator:
    def __init__(self):
        print("Initializing Local LLM via Ollama (llama3.2:1b)...")
        # Uses the new dedicated langchain-ollama package
        self.llm = OllamaLLM(model="llama3.2:1b", temperature=0.3)
        
        # The System Prompt
        self.prompt = PromptTemplate(
            input_variables=["brand", "context", "query"],
            template="""You are a professional customer support agent for {brand}. 
            A customer has reached out with the following issue:
            "{query}"
            
            Here is how your company resolved similar issues in the past:
            {context}
            
            Based strictly on these past solutions, draft a polite, helpful, and concise response to the customer. 
            Do not invent company policies, links, or phone numbers that are not in the provided context.
            
            Draft Response:"""
        )
        
        self.chain = self.prompt | self.llm
        
    def generate_draft(self, query, brand, retrieved_docs):
        if not retrieved_docs:
            return "No historical context found. Flagging for Human Agent."
            
        # Combine the retrieved past solutions into one text block for the LLM
        context = "\n\n".join([
            f"Past Solution: {doc.metadata.get('outbound_text')}" 
            for doc in retrieved_docs
        ])
        
        print(f"\n--- Synthesizing Final Response for {brand.capitalize()} (Running Locally) ---")
        
        # This will run entirely locally on your CPU
        response = self.chain.invoke({
            "brand": brand,
            "context": context,
            "query": query
        })
        
        return response

if __name__ == "__main__":
    # 1. Initialize both agents only once at startup
    triage = TriageAgent()
    generator = ResponseGenerator()
    
    print("\n" + "="*50)
    print("Agentic Support Triage RAG - Interactive CLI")
    print("Type 'exit' or 'quit' to stop the agent.")
    print("="*50)
    
    # 2. Start the interactive loop
    while True:
        # Get live input from the terminal
        user_query = input("\n[Customer] Enter query: ")
        
        # Check if the user wants to stop
        if user_query.strip().lower() in ['exit', 'quit']:
            print("\nShutting down agent. Goodbye!")
            break
            
        # Skip empty inputs
        if not user_query.strip():
            continue
            
        # 3. Route it
        domain, is_urgent = triage.route_query(user_query)
        
        # 4. Retrieve past solutions from the vector DB
        retrieved_docs = triage.retrieve_solutions(user_query, domain)
        
        # 5. Extract the brand name
        brand_name = retrieved_docs[0].metadata.get('brand', domain) if retrieved_docs else domain
        
        # 6. Generate the final output
        if is_urgent:
            print("\n[SYSTEM] Query marked as HIGH URGENCY. Bypassing AI auto-reply. Routing to Human Dashboard.")
        else:
            final_answer = generator.generate_draft(user_query, brand_name, retrieved_docs)
            print("\n[Agent] FINAL AI DRAFT:")
            print(final_answer)