import requests
import json
import http.client
import time
import re

        
def web_search(query, config):
    if not query:
        raise ValueError("Search query cannot be empty")
    if config['search_engine'] == 'google':
        return serper_google_search(
            query=query,
            serper_api_key=config['serper_api_key'],
            top_k=config['search_top_k'],
            region=config['search_region'],
            lang=config['search_lang']
        )
    elif config['search_engine'] == 'bing':
        return azure_bing_search(
            query=query,
            subscription_key=config['azure_bing_search_subscription_key'],
            mkt=config['azure_bing_search_mkt'],
            top_k=config['search_top_k']
        )
    elif config['search_engine'] == 'grt':
        return grt_search(
            query=query,
            top_k=config['search_top_k'],
            region=config['search_region'],
            lang=config['search_lang']
        )


def azure_bing_search(query, subscription_key, mkt, top_k, depth=0):
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    results = []

    try:
        response = requests.get("", headers=headers, params=params)
        json_response = response.json()
        for e in json_response['webPages']['value']:
            results.append({
                "title": e['name'],
                "link": e['url'],
                "snippet": e['snippet']
            })
    except Exception as e:
        print(f"Bing search API error: {e}")
        if depth < 1024:
            time.sleep(1)
            return azure_bing_search(query, subscription_key, mkt, top_k, depth+1)
    return results


from serpapi import GoogleSearch
def serper_google_search(
        query, 
        serper_api_key,
        top_k,
        region,
        lang,
        depth=0
    ):
    try:
        params = {
          "api_key": serper_api_key,
          "engine": "google",
          "q": query,
          "location": "Austin, Texas, United States",
          "google_domain": "google.com",
          "gl": region,
          "hl": lang
        }
        
        search = GoogleSearch(params)
        data = search.get_dict()
        if 'organic_results' not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic_results"]
            print("search success")
            return results
    except Exception as e:
        print(f"Serper search API error: {e}")
        if depth < 512:
            time.sleep(1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print("search failed")
    return []


def grt_search(
        query, 
        top_k,
        region,
        lang,
        depth=0
    ):
    """
    Perform a web search using the bing google quark etc.
    """
    url = ''
    headers = {'content-type': 'application/json'}
    env = "PROD"

    data = {
        "functionName": "",
        "env": env,
        "params": {
            "userId":"200324",
            "source":"smart_center",
            "query":query,
            "count":str(top_k)
        }
    }
    search_response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if search_response.status_code != 200:
        raise Exception(f"Error occurred while searching: {search_response.text}")
    try:
        search_results = json.loads(search_response.json()['data'])
        if len(search_results) == 0:
            return []
        else:
            formatted_results = ""
            results = []
            i = 0
            formatted_results = [x for x in search_results if x['searchStrategy'] in ['bing','google']]
            for e in formatted_results:
                e['title'] = e['sourceTitle']
                del e['sourceTitle']
                e['snippet'] = e['abstractText']
                del e['abstractText']
                e['link'] = e['sourceUrl']
                del e['sourceUrl']
        return formatted_results
    except Exception as e:
        print(f"grt search API error: {e}")
        print("search failed")
        return []



def extract_full_hostname(url):
    match = re.search(r'(?:https?://)?([^:/\s]+)', url)
    if match:
        hostname = match.group(1)
        return hostname.split(':')[0]
    return None

def serper_google_search_old(
        query, 
        serper_api_key,
        top_k,
        region,
        lang,
        depth=0
    ):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        print('data',data)
        if not data:
            raise Exception("The google search API is temporarily unavailable, please try again later.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic"]
            print("search success")
            return results
    except Exception as e:
        # print(f"Serper search API error: {e}")
        if depth < 512:
            time.sleep(1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print("search failed")
    return []


if __name__ == "__main__":
    print(serper_google_search("test", "your serper key",1,"us","en"))