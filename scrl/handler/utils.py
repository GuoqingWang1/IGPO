from typing import List, Dict, Any, Optional
from openai import OpenAI
import re
from urllib.parse import urlparse
import time

def extract_url_root_domain(url):
    """
    Extract the root domain from a URL.
    Examples:
    - https://www.example.com/path -> example.com
    - sub.example.co.uk -> example.co.uk
    """
    # Ensure the URL contains a protocol; if not, add one
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Parse the URL using urlparse
    parsed = urlparse(url).netloc
    if not parsed:
        parsed = url
        
    # Remove the port number (if present)
    parsed = parsed.split(':')[0]
    
    # Split the domain parts
    parts = parsed.split('.')
    
    # Handle special second-level domains, such as .co.uk, .com.cn, etc.
    if len(parts) > 2:
        if parts[-2] in ['co', 'com', 'org', 'gov', 'edu', 'net']:
            if parts[-1] in ['uk', 'cn', 'jp', 'br', 'in']:
                return '.'.join(parts[-3:])
    
    # Return the main domain (last two parts)
    return '.'.join(parts[-2:])

def get_clean_content(line):
    clean_line = re.sub(r'^[\*\-•#\d\.]+\s*', '', line).strip()
    clean_line = re.sub(r'^[\'"]|[\'"]$', '', clean_line).strip()
    if (clean_line.startswith('"') and clean_line.endswith('"')) or \
    (clean_line.startswith("'") and clean_line.endswith("'")):
        clean_line = clean_line[1:-1]
    return clean_line

def get_content_from_tag(content, tag, default_value=None):
    # Explanation:
    # 1) (.*?) Lazy match to capture as few characters as possible
    # 2) (?=(</tag>|<\w+|$)) Uses lookahead, meaning matching stops when followed by </tag>, 
    #    a tag starting with a word character, or the end of the text
    # 3) re.DOTALL allows the dot '.' to match newline characters
    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return default_value


def get_response_from_llm(
        messages: List[Dict[str, Any]],
        client: OpenAI,
        model: str,
        stream: Optional[bool] = False,
        temperature: Optional[float] = 0.6,
        depth: int = 0
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream
        )
        if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
            content = response.choices[0].message.content
        return {
            "content": content.strip()
        }
    except Exception as e:
        print(f"LLM API error: {e}")
        if "Input data may contain inappropriate content" in str(e):
            return {
                "content": ""
            }
        if "Error code: 400" in str(e):
            return {
                "content": ""
            }
        if depth < 512:
            time.sleep(1)
            return get_response_from_llm(messages=messages, client=client, model=model, stream=stream, temperature=temperature, depth=depth+1)
        raise e