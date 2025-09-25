
QUICK_SUMMARY_PROMPT = """You are an AI assistant analyzing webpage content to determine if it's helpful for answering a user's question. Given:

1. User query: {user_query}
2. Search query: {search_query}
3. Webpage content: {first_page_fetch_res}

Evaluate if this webpage contains useful information for answering the user's question or search query.

Think through:
1. What key information does the webpage contain?
2. How does this information relate to the user's question or search query?
3. Is the content sufficient and relevant to help answer the query?

Provide your analysis in this format:
<helpful>yes/no</helpful>
<summary>If helpful: Concise summary of relevant information that helps answer the query</summary>"""
