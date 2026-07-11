import os
# Configure OpenBLAS and other thread pools to use 1 thread to prevent container memory errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import json
from dotenv import load_dotenv
from groq import Groq
from router import HybridRouter

# Load environment variables
load_dotenv()

class LenDenClubAutomationPipeline:
    def __init__(self):
        # Initialize Hybrid Router
        self.router = HybridRouter()
        
        # Setup Groq Client
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.is_simulated = False
        
        if not self.groq_api_key or "your_groq_api_key" in self.groq_api_key:
            print("\n[NOTICE] GROQ_API_KEY not configured or placeholder detected. Running pipeline in SIMULATED LLM Mode.")
            self.is_simulated = True
            self.groq_client = None
        else:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("\n[SUCCESS] Groq API client initialized successfully.")
            except Exception as e:
                print(f"\n[WARNING] Failed to initialize Groq client: {e}. Falling back to SIMULATED Mode.")
                self.is_simulated = True
                self.groq_client = None

    def execute_llm_generation(self, prompt):
        """Calls Groq Llama 3.3 model or simulates the output if API key is missing."""
        if self.is_simulated:
            return self._simulate_llama_response(prompt)
            
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional customer support representative for LenDenClub. Always draft emails using clear, professional language, and strictly follow the provided business rules and template structures."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=1024
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}. Falling back to simulation.")
            return self._simulate_llama_response(prompt)

    def _simulate_llama_response(self, prompt):
        """Simulates response generation for verification purposes when offline."""
        if "Draft Template:" in prompt:
            customer_name_match = re.search(r"Customer Name:\s*(.*?)\n", prompt)
            customer_name = customer_name_match.group(1) if customer_name_match else "Customer"
            
            template_match = re.search(r"Draft Template:\s*\n(.*?)(?=\n\nOriginal Customer|$)", prompt, re.DOTALL)
            template_text = template_match.group(1) if template_match else ""
            
            filled_template = template_text.replace("{{customer_name}}", customer_name)
            filled_template = re.sub(r"\{\{.*?\}\}", "[N/A]", filled_template)
            
            simulated_response = f"[SIMULATED LLM RESPONSE (Llama 3.3)]\n\n{filled_template}\n\nNote: If there are additional details in the customer inquiry that weren't in the template, a live LLM would synthesize those here using the provided Knowledge Base context."
            return simulated_response
        else:
            return "[SIMULATED LLM RESPONSE (Llama 3.3)]\n\nDear Customer,\n\nThank you for reaching out to LenDenClub. As you are a new customer, we don't have an active account under your email/ID. To apply for our InstaMoney personal loans (salaried individuals, min INR 20k income) or to start P2P lending investments via IM+, please download the LenDenClub app and complete your quick KYC process.\n\nBest regards,\nLenDenClub Customer Support Team"

    def run_pipeline(self, customer_id, email_query):
        """Runs the entire automation pipeline for a single email query."""
        # Step 1: Hybrid Routing (Templates Search)
        route_result = self.router.route_email(customer_id, email_query)
        
        # Step 2: Context Retrieval & Prompt Construction
        if route_result["rag_required"]:
            segment = route_result["product_segment"]
            customer_name = route_result["customer_name"]
            template = route_result["template"]
            
            # Semantically query the knowledge base collection in Milvus Lite
            print("Querying Milvus Lite database for relevant knowledge base context...")
            kb_match = self.router.match_kb_similarity(email_query, segment)
            if kb_match:
                print(f"KB Route: SEMANTIC RETRIEVAL -> Section: {kb_match['section_title']} (Distance/Score: {kb_match['score']:.4f})")
                kb_context = f"Relevant KB Section ({kb_match['product_segment']}): {kb_match['section_title']}\n{kb_match['text_payload']}"
            else:
                print("No relevant KB context resolved semantically. Using fallback.")
                kb_context = "No specific context retrieved."
            
            # Format raw template payload placeholders for the prompt
            raw_template_payload = template["text_payload"]
            
            prompt = f"""You are an advanced customer support AI agent for LenDenClub.
You are responding to an email from a customer.

Customer Name: {customer_name}
Product Segment: {segment}

Knowledge Base Context:
{kb_context}

Draft Template:
{raw_template_payload}

Original Customer Inquiry:
{email_query}

Write a personalized, clear, and professional email response to the customer. Follow the rules from the Knowledge Base and use the Draft Template as a guide. Ensure all specific questions are answered and details are accurate. Do not include placeholders like {{{{...}}}} in the final output. Replace them with the actual customer details.
"""
        else:
            # Non-RAG Route (New Customer)
            prompt = f"""You are an advanced customer support AI agent for LenDenClub.
A new customer is inquiring. We do not have their profile or segment information in our RAG database yet.

Original Customer Inquiry:
{email_query}

Write a polite, professional response answering their query generally based on standard guidelines. Suggest they sign up or complete KYC verification to get personalized details.
"""

        # Step 3: LLM Response Generation
        print("\n--- Generating Email Response ---")
        response = self.execute_llm_generation(prompt)
        print(f"Generated Response:\n{response}")
        print("="*60)
        
        return {
            "routing": route_result,
            "prompt": prompt,
            "response": response
        }

# --- TEST SUITE RUNNER ---
if __name__ == "__main__":
    pipeline = LenDenClubAutomationPipeline()
    
    # 7 diverse mock test queries mapping to different routing scenarios
    test_queries = [
        # Scenario 1: New Customer (Direct LLM)
        {
            "customer_id": "CUST999", # New ID
            "email": "Hi, I am looking to invest some capital. Do you support P2P lending portfolios for individual retail investors?"
        },
        # Scenario 2: Existing IM Customer - Keyword Match (im_eligibility_cibil)
        {
            "customer_id": "CUST001", # Existing IM customer (Aarav Sharma)
            "email": "Dear team, I want to apply for a personal loan but I am worried about my credit score. What is the minimum CIBIL score required for my application?"
        },
        # Scenario 3: Existing IM+ Customer - Keyword Match (im_plus_lock_in)
        {
            "customer_id": "CUST002", # Existing IM+ customer (Priya Patel)
            "email": "Hello, I recently activated an IM+ investment portfolio. Can you confirm the lock-in period details? How long is my capital locked?"
        },
        # Scenario 4: Existing IM Customer - Vector Fallback Match (im_disbursement_time)
        {
            "customer_id": "CUST003", # Existing IM customer (Vikram Singh)
            "email": "Hey support, I signed the loan contract and set up the bank mandate successfully. When will the money actually hit my account? How long does the cash transfer process usually take?"
        },
        # Scenario 5: Existing IM Customer - Segment Filtration security rule test
        {
            "customer_id": "CUST005", # Existing IM customer
            "email": "Hi, I heard about your IM+ premium investment plans. Can you explain the expected returns of the Aggressive risk portfolio?"
        },
        # Scenario 6: Existing IM+ Customer - Vector Fallback Match for IM query (im_late_payment_penalty)
        {
            "customer_id": "CUST008", # Existing IM+ customer (Neha Gupta)
            "email": "Can you explain the penalties for late repayment on personal loans? I have a friend who missed their EMI date."
        },
        # Scenario 7: Existing Customer - BOTH Category Keyword Match (both_profile_update)
        {
            "customer_id": "CUST009", # Existing IM customer (Arjun Malhotra)
            "email": "Dear support, I need to update my email address associated with my account. How can I modify these profile details?"
        }
    ]
    
    print("\n" + "="*80)
    print("STARTING LENDENCLUB AI-POWERED EMAIL AUTOMATION TEST PIPELINE")
    print("="*80)
    
    for i, q in enumerate(test_queries, 1):
        print(f"\n[TEST CASE {i}]")
        pipeline.run_pipeline(q["customer_id"], q["email"])
        
    print("\n" + "="*80)
    print("TEST PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
