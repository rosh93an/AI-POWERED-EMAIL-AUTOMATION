import os
# Configure OpenBLAS and other thread pools to use 1 thread to prevent container memory errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Suppress verbose gRPC keepalive logs and connection warnings in the console
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""


import json
import numpy as np
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class HybridRouter:
    def __init__(self):
        # Database and collection configurations
        self.db_file = os.getenv("MILVUS_DB_FILE", "lenden_automation.db")
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME", "email_templates")
        self.kb_collection_name = os.getenv("MILVUS_KB_COLLECTION_NAME", "knowledge_base")
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        # Load local JSON configurations
        self.customer_profiles_path = "customer_profiles.json"
        self.templates_json_path = os.getenv("TEMPLATES_JSON_FILE", "customer_templates.json")
        
        # Load data stores
        with open(self.customer_profiles_path, "r") as f:
            self.customer_profiles = json.load(f)
        with open(self.templates_json_path, "r") as f:
            self.templates = json.load(f)
            
        # Instantiate MilvusClient (lite version running locally)
        print(f"Connecting to Milvus Lite database file: {self.db_file}")
        self.client = MilvusClient(self.db_file)
        
        # Lazy load the SentenceTransformer model
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully.")
        
        # Ensure collections are setup
        self._ensure_collection_setup()
        
        # Load collections into memory (essential for query/search in new sessions)
        print(f"Loading collections '{self.collection_name}' and '{self.kb_collection_name}' into memory...")
        self.client.load_collection(self.collection_name)
        self.client.load_collection(self.kb_collection_name)

    def _ensure_collection_setup(self):
        """Checks if both collections exist, if not initializes them."""
        has_templates = self.client.has_collection(self.collection_name)
        has_kb = self.client.has_collection(self.kb_collection_name)
        
        if not has_templates or not has_kb:
            print("One or more collections missing. Re-initializing database...")
            self.initialize_vector_db()
        else:
            print("Both Milvus collections ('email_templates' and 'knowledge_base') verified.")

    def initialize_vector_db(self):
        """Drops existing database folder to bypass Windows Milvus Lite FileExistsError bugs, and populates fresh collections."""
        print("Resetting local Milvus Lite database files to avoid Windows file locks...")
        self.client.close()
        
        import shutil
        if os.path.exists(self.db_file):
            try:
                shutil.rmtree(self.db_file)
                print(f"Database directory '{self.db_file}' deleted successfully.")
            except Exception as e:
                print(f"Warning: Could not delete database directory: {e}. Trying to clean up files inside it...")
                for filename in os.listdir(self.db_file):
                    file_path = os.path.join(self.db_file, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as ex:
                        print(f"Failed to delete {file_path}: {ex}")
                        
        # Re-initialize MilvusClient
        self.client = MilvusClient(self.db_file)
        
        # 1. Setup email_templates collection
        print(f"Creating collection '{self.collection_name}'...")
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=384  # Matches all-MiniLM-L6-v2 embedding dimension
        )
        
        print("Embedding templates and inserting into Milvus...")
        data_to_insert = []
        for idx, template in enumerate(self.templates):
            keywords_str = ", ".join(template["keywords"])
            text_to_embed = f"Product: {template['product_segment']}. Keywords: {keywords_str}. Description: {template['description']}. Template Payload: {template['text_payload']}"
            embedding = self.model.encode(text_to_embed).tolist()
            
            data_to_insert.append({
                "id": idx,
                "vector": embedding,
                "template_id": template["template_id"],
                "product_segment": template["product_segment"],
                "text_payload": template["text_payload"],
                "description": template["description"],
                "keywords": template["keywords"]
            })
            
        self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )
        print(f"Loaded {len(data_to_insert)} templates into collection '{self.collection_name}'.")

        # 2. Setup knowledge_base collection
        kb_file_path = os.getenv("KNOWLEDGE_BASE_FILE", "knowledge_base.txt")
        if os.path.exists(kb_file_path):
            print(f"Parsing knowledge base file: {kb_file_path}...")
            with open(kb_file_path, "r", encoding="utf-8") as f:
                kb_content = f.read()
            
            import re
            # Split by product block
            segments = re.split(r"\[PRODUCT:\s*(IM|IM\+)\s*-.*?\]", kb_content)
            
            kb_data_to_insert = []
            chunk_idx = 0
            
            for i in range(1, len(segments), 2):
                prod_seg = segments[i]
                block_text = segments[i+1]
                
                # Extract numbered sections
                sections = re.findall(r"(\d+\.\s+[^:\n]+:)\s*\n(.*?)(?=\n\d+\.\s+|$)", block_text, re.DOTALL)
                
                # Check for intro block
                intro_match = re.search(r"^(.*?)(?=\n1\.\s+)", block_text, re.DOTALL)
                if intro_match and intro_match.group(1).strip():
                    intro_text = intro_match.group(1).strip()
                    kb_data_to_insert.append({
                        "id": chunk_idx,
                        "vector": self.model.encode(f"Product: {prod_seg}. Section: Overview. Content: {intro_text}").tolist(),
                        "product_segment": prod_seg,
                        "section_title": "Overview",
                        "text_payload": intro_text
                    })
                    chunk_idx += 1
                    
                for title, body in sections:
                    body_text = body.strip()
                    kb_data_to_insert.append({
                        "id": chunk_idx,
                        "vector": self.model.encode(f"Product: {prod_seg}. Section: {title}. Content: {body_text}").tolist(),
                        "product_segment": prod_seg,
                        "section_title": title.strip(),
                        "text_payload": body_text
                    })
                    chunk_idx += 1
            
            if kb_data_to_insert:
                print(f"Creating collection '{self.kb_collection_name}'...")
                self.client.create_collection(
                    collection_name=self.kb_collection_name,
                    dimension=384
                )
                self.client.insert(
                    collection_name=self.kb_collection_name,
                    data=kb_data_to_insert
                )
                print(f"Loaded {len(kb_data_to_insert)} knowledge base chunks into collection '{self.kb_collection_name}'.")
        else:
            print(f"Warning: {kb_file_path} not found. Skipping KB database population.")

    def resolve_customer_identity(self, customer_id):
        """Resolves customer status and product segment."""
        for customer in self.customer_profiles:
            if customer["customer_id"] == customer_id:
                return {
                    "is_existing": True,
                    "name": customer["name"],
                    "product_segment": customer["product_segment"],
                    "email": customer["email"]
                }
        return {
            "is_existing": False,
            "name": "New Customer",
            "product_segment": None,
            "email": None
        }

    def match_keywords(self, email_body, product_segment):
        """Scans email body for template keywords, enforcing product segment rules."""
        email_body_lower = email_body.lower()
        best_match = None
        highest_keyword_count = 0
        
        for template in self.templates:
            # Enforce Segment Filtering Rules
            t_segment = template["product_segment"]
            if product_segment == "IM" and t_segment == "IM+":
                continue
                
            # Count keyword occurrences
            match_count = 0
            for keyword in template["keywords"]:
                if f" {keyword.lower()} " in f" {email_body_lower} " or email_body_lower.startswith(keyword.lower()) or email_body_lower.endswith(keyword.lower()):
                    match_count += 1
                    
            if match_count > highest_keyword_count:
                highest_keyword_count = match_count
                best_match = template
                
        if highest_keyword_count > 0:
            return best_match
        return None

    def match_vector_similarity(self, email_body, product_segment):
        """Generates query embedding and searches templates in Milvus."""
        query_vector = self.model.encode(email_body).tolist()
        
        # Enforce Segment Filtering Rules
        if product_segment == "IM":
            filter_expr = 'product_segment in ["IM", "BOTH"]'
        else:
            filter_expr = 'product_segment in ["IM", "IM+", "BOTH"]'
            
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=1,
            filter=filter_expr,
            output_fields=["template_id", "product_segment", "text_payload", "description", "keywords"]
        )
        
        if results and len(results[0]) > 0:
            top_hit = results[0][0]
            entity = top_hit["entity"]
            return {
                "template_id": entity["template_id"],
                "product_segment": entity["product_segment"],
                "text_payload": entity["text_payload"],
                "description": entity["description"],
                "keywords": entity["keywords"],
                "score": top_hit["distance"]
            }
        return None

    def match_kb_similarity(self, email_body, product_segment):
        """Generates query embedding and searches knowledge base chunks in Milvus."""
        query_vector = self.model.encode(email_body).tolist()
        
        # Enforce Segment Filtering Rules
        if product_segment == "IM":
            filter_expr = 'product_segment in ["IM", "BOTH"]'
        else:
            filter_expr = 'product_segment in ["IM", "IM+", "BOTH"]'
            
        results = self.client.search(
            collection_name=self.kb_collection_name,
            data=[query_vector],
            limit=1,
            filter=filter_expr,
            output_fields=["section_title", "product_segment", "text_payload"]
        )
        
        if results and len(results[0]) > 0:
            top_hit = results[0][0]
            entity = top_hit["entity"]
            return {
                "section_title": entity["section_title"],
                "product_segment": entity["product_segment"],
                "text_payload": entity["text_payload"],
                "score": top_hit["distance"]
            }
        return None

    def route_email(self, customer_id, email_body):
        """Main entry point for routing emails, resolving identity, and searching templates."""
        print(f"\n--- Routing Email for Customer ID: {customer_id} ---")
        print(f"Email Body: \"{email_body[:80]}...\"")
        
        # 1. Identity Resolution
        identity = self.resolve_customer_identity(customer_id)
        if not identity["is_existing"]:
            print("Status: NEW CUSTOMER -> Non-RAG Route (Direct LLM Response)")
            return {
                "customer_status": "NEW",
                "customer_name": identity["name"],
                "product_segment": None,
                "rag_required": False,
                "routing_type": "DIRECT_LLM",
                "template": None,
                "reason": "New customer ID"
            }
            
        customer_segment = identity["product_segment"]
        print(f"Status: EXISTING CUSTOMER ({identity['name']}) -> Segment: {customer_segment}")
        
        # 2. Keyword Match
        keyword_template = self.match_keywords(email_body, customer_segment)
        if keyword_template:
            print(f"Route: KEYWORD MATCH -> Template ID: {keyword_template['template_id']}")
            return {
                "customer_status": "EXISTING",
                "customer_name": identity["name"],
                "product_segment": customer_segment,
                "rag_required": True,
                "routing_type": "KEYWORD",
                "template": keyword_template,
                "reason": f"Keywords matched template {keyword_template['template_id']}"
            }
            
        # 3. Vector Similarity Fallback
        print("No keyword match found. Falling back to semantic vector similarity match...")
        vector_template = self.match_vector_similarity(email_body, customer_segment)
        if vector_template:
            print(f"Route: VECTOR SIMILARITY FALLBACK -> Template ID: {vector_template['template_id']} (Distance/Score: {vector_template['score']:.4f})")
            return {
                "customer_status": "EXISTING",
                "customer_name": identity["name"],
                "product_segment": customer_segment,
                "rag_required": True,
                "routing_type": "VECTOR_SIMILARITY",
                "template": vector_template,
                "reason": f"Vector match: {vector_template['template_id']} with distance {vector_template['score']:.4f}"
            }
            
        print("Route Fallback: No template resolved.")
        return {
            "customer_status": "EXISTING",
            "customer_name": identity["name"],
            "product_segment": customer_segment,
            "rag_required": False,
            "routing_type": "FALLBACK_DIRECT_LLM",
            "template": None,
            "reason": "No keyword or semantic match found"
        }
