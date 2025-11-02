# routing_logic.py
# Purpose: Determines customer identity and product scope based on Freshdesk data.

import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def determine_customer_scope(freshdesk_data: dict) -> dict:
    """
    Analyzes incoming Freshdesk data (customer_id, ticket_id, products)
    to set the scope for RAG and generation.

    Args:
        freshdesk_data: Dictionary containing customer inquiry details.

    Returns:
        A dict with the routing decision and product context.
    """
    customer_id = freshdesk_data.get('customer_id')
    ticket_id = freshdesk_data.get('ticket_id')
    
    # 1. Identity Resolution (Existing vs. New)
    if customer_id and ticket_id:
        is_existing = True
        routing_mode = 'RAG_FULL'
        logging.info(f"Customer {customer_id} is existing, requires RAG.")
    else:
        is_existing = False
        routing_mode = 'GEN_SIMPLE'
        logging.info(f"Customer is new, routing for simple LLM generation.")
    
    # 2. Product Context Resolution (IM vs IM+) - The Complex Business Rule
    # NOTE: This is a placeholder for the complex business logic (IM+ can be IM, etc.)
    customer_products = freshdesk_data.get('customer_products', [])
    
    if 'IM+' in customer_products:
        product_scope = ['IM+']
        # Rule: IM+ customer can implicitly access IM info if needed for context
        if 'IM' not in customer_products:
             product_scope.append('IM') 
    elif 'IM' in customer_products:
        product_scope = ['IM']
    else:
        product_scope = []

    return {
        'is_existing': is_existing,
        'routing_mode': routing_mode,
        'product_scope': product_scope
    }

# Example Usage:
if __name__ == '__main__':
    # Scenario 1: Existing customer with IM+ product
    query_data_existing = {
        'customer_id': 'CUST123',
        'ticket_id': 'TICK456',
        'query_text': 'What is the refund policy for my IM+ plan?',
        'customer_products': ['IM+']
    }
    
    print("--- Scenario 1 ---")
    print(determine_customer_scope(query_data_existing))

    # Scenario 2: New customer query (no existing ID/ticket)
    query_data_new = {
        'customer_id': None,
        'ticket_id': None,
        'query_text': 'How do I start an IM plan?',
        'customer_products': []
    }
    
    print("\n--- Scenario 2 ---")
    print(determine_customer_scope(query_data_new))
