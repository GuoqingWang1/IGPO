from web_search_agent.web_search_agent import WebSearchAgent
from reading_agent.reading_agent import ReadingAgent
import yaml
from openai import OpenAI
from typing import List
from webpage import SearchResultInfo
import json
from agent_action import *
import time
from types import SimpleNamespace
import concurrent.futures
from web_search_agent.search.search_api import web_search
import threading
import time
import os
from tqdm import tqdm
import requests

class Handler:
    def __init__(self, agent_config, client, handler_config):
        self.web_search_agent = WebSearchAgent(config=agent_config, client=client)
        self.reading_agent = ReadingAgent(config=agent_config, client=client)
        self.id_to_context: dict[int, list[ActionInfo]] = {}
        self.agent_config = agent_config
        self.client = client
        self.handler_config = handler_config
        self.api_result_dict = {}
        if os.path.exists(self.agent_config["query_save_path"]):
            with open(self.agent_config["query_save_path"], 'r', encoding='utf-8') as f:
                self.api_result_dict = json.load(f)
        self.query_save_path = self.agent_config["query_save_path"]
        self.id_to_context_lock = threading.Lock()
    
    def search_and_add_to_dict(self, search_query, cur_api_result_dict, cur_api_result_dict_lock):
        try:
            organic = web_search(search_query, self.agent_config)
            with cur_api_result_dict_lock:
                cur_api_result_dict[search_query] = {
                    'timestamp': time.time(),
                    'organic': organic
                }
        except Exception as e:
            print(f"Error in search_and_add_to_dict for query '{search_query}': {str(e)}")
    
    def handle_execution_api(self, query_contents: List[dict]):
        print("Start processing function calls")
        cur_api_result_dict = {}
        cur_api_result_dict_lock = threading.Lock()
        cache_hit = 0
        total_search_call = 0
        for query_content in query_contents:
            if 'tool_call' not in query_content:
                continue
            if 'name' not in query_content['tool_call'] or query_content['tool_call']['name'] != 'web_search':
                continue
            if 'arguments' not in query_content['tool_call']:
                continue
            if 'query' not in query_content['tool_call']['arguments']:
                continue
            query_list = query_content['tool_call']['arguments']['query']
            if not isinstance(query_list, list):
                continue
            total_search_call += len(query_list)
            for query in query_list:
                if type(query) != str:
                    continue
                if query in self.api_result_dict and len(self.api_result_dict[query]['organic']) > 0 and (time.time() - self.api_result_dict[query]['timestamp'] <= 60 * 60 * 24 * 7):
                    cache_hit += 1
                    continue
                cur_api_result_dict[query] = {
                    "timestamp": time.time(),
                    "organic": []
                }
        if total_search_call > 0:
            print(f"Total calls this round:{total_search_call}, cache hits: {cache_hit}, hit rate:{cache_hit/total_search_call}", flush=True)
        start_time = time.time()
        api_future_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as api_executor:
            for search_query in cur_api_result_dict:
                api_future = api_executor.submit(self.search_and_add_to_dict, search_query, cur_api_result_dict, cur_api_result_dict_lock)
                api_future_list.append(api_future)
        print(f"Search time:{time.time()-start_time}", flush=True)
        
        for api_future in concurrent.futures.as_completed(api_future_list):
            api_future.result()
        
        for key in cur_api_result_dict:
            self.api_result_dict[key] = cur_api_result_dict[key]
            
        with open(self.agent_config["query_save_path"], 'w', encoding='utf-8') as f:
            json.dump(self.api_result_dict, f, indent=4, ensure_ascii=False)
        print("Cache saved", flush=True)
        start_time = time.time()
        future_to_content = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as handle_executor:
            for query_content in query_contents:
                future = handle_executor.submit(self.handle_single_query, query_content, self.api_result_dict)
                future_to_content.append(future)
            # Wait for all tasks to finish
        for i, future in enumerate(future_to_content):
            query_contents[i]["content"] = future.result()
        print(f"Crawling/reading webpages took:{time.time()-start_time}")
        print("Function call processing finished")
        return query_contents

    def handle_execution(self):
        while True:
            has_query = False
            while not has_query:
                with open(self.handler_config.signal_writing_file, 'r', encoding='utf-8') as f:
                    signal_contents = json.load(f)
                
                if signal_contents['signal'] == self.handler_config.QUERY_SIGNAL:
                    has_query = True
                else:
                    time.sleep(10)
            with open(self.handler_config.signal_writing_file, 'r', encoding='utf-8') as f:
                signal_contents = json.load(f)
            assert signal_contents['signal'] == self.handler_config.QUERY_SIGNAL
            
            with open(self.handler_config.data_writing_file, 'r', encoding="utf-8") as f:
                query_contents = json.load(f)
            
            cur_api_result_dict = {}
            cur_api_result_dict_lock = threading.Lock()
            cache_hit = 0
            total_search_call = 0
            for query_content in query_contents:
                if 'tool_call' not in query_content:
                    continue
                if 'name' not in query_content['tool_call'] or query_content['tool_call']['name'] != 'web_search':
                    continue
                if 'arguments' not in query_content['tool_call']:
                    continue
                if 'query' not in query_content['tool_call']['arguments']:
                    continue
                query_list = query_content['tool_call']['arguments']['query']
                if not isinstance(query_list, list):
                    continue
                total_search_call += len(query_list)
                for query in query_list:
                    if type(query) != str:
                        continue
                    if query in self.api_result_dict and len(self.api_result_dict[query]['organic']) > 0 and (time.time() - self.api_result_dict[query]['timestamp'] <= 60 * 60 * 24 * 7):
                        cache_hit += 1
                        continue
                    cur_api_result_dict[query] = {
                        "timestamp": time.time(),
                        "organic": []
                    }
            if total_search_call > 0:
                print(f"Total calls this round:{total_search_call}, cache hits: {cache_hit}, hit rate: {cache_hit/total_search_call}", flush=True)
            start_time = time.time()
            api_future_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as api_executor:
                for search_query in cur_api_result_dict:
                    api_future = api_executor.submit(self.search_and_add_to_dict, search_query, cur_api_result_dict, cur_api_result_dict_lock)
                    api_future_list.append(api_future)
            print(f"Search time:{time.time()-start_time}", flush=True)
            
            for api_future in concurrent.futures.as_completed(api_future_list):
                api_future.result()
            
            for key in cur_api_result_dict:
                self.api_result_dict[key] = cur_api_result_dict[key]
                
            with open(self.agent_config["query_save_path"], 'w', encoding='utf-8') as f:
                json.dump(self.api_result_dict, f, indent=4, ensure_ascii=False)
            print("Cache saved", flush=True)
            start_time = time.time()
            future_to_content = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as handle_executor:
                for query_content in query_contents:
                    future = handle_executor.submit(self.handle_single_query, query_content, self.api_result_dict)
                    future_to_content.append(future)
                # Wait for all tasks to finish
            for future in concurrent.futures.as_completed(future_to_content):
                future.result()
            print(f"Crawling/reading webpages took:{time.time()-start_time}")

            with open(self.handler_config.data_writing_file, 'w', encoding="utf-8") as f:
                json.dump(query_contents, f, indent=4, ensure_ascii=False)
            
            with open(self.handler_config.data_writing_file, 'r', encoding='utf-8') as f:
                query_contents_check = json.load(f)
            assert query_contents == query_contents_check

            with open(self.handler_config.signal_writing_file, 'w', encoding='utf-8') as f:
                json.dump({'signal': self.handler_config.RESPONSE_SIGNAL}, f, indent=4, ensure_ascii=False)
            print("Function call processing finished")
            
    def foward_to_server(self):
        server_url_list = self.agent_config['server_url_list']
        server_cnt = len(server_url_list)

        while True:
            has_query = False
            while not has_query:
                with open(self.handler_config.signal_writing_file, 'r', encoding='utf-8') as f:
                    signal_contents = json.load(f)
                
                if signal_contents['signal'] == self.handler_config.QUERY_SIGNAL:
                    has_query = True
                else:
                    time.sleep(10)
            print("Start processing function calls")
            with open(self.handler_config.signal_writing_file, 'r', encoding='utf-8') as f:
                signal_contents = json.load(f)
            assert signal_contents['signal'] == self.handler_config.QUERY_SIGNAL
            
            with open(self.handler_config.data_writing_file, 'r', encoding="utf-8") as f:
                query_contents = json.load(f)
            if len(query_contents) == 0:
                handler_result_list = []
            else:
                batch_cnt_per_server = query_contents[0]["total_number"] // server_cnt + 1
                server_query_contents_list = [[] for _ in range(server_cnt)]
                for query_content in query_contents:
                    server_query_contents_list[query_content['idx'] // batch_cnt_per_server].append(query_content)

                with concurrent.futures.ThreadPoolExecutor(max_workers=server_cnt) as api_executor:
                    api_future_list = []
                    for server_idx in range(server_cnt):
                        api_future = api_executor.submit(self.post_request, server_url_list[server_idx], server_query_contents_list[server_idx],server_idx)
                        api_future_list.append(api_future)
                handler_result_list = []
                for i, future in enumerate(api_future_list):
                    handler_result_list += future.result()
            print(len(handler_result_list), flush=True)
            with open(self.handler_config.data_writing_file, 'w', encoding="utf-8") as f:
                json.dump(handler_result_list, f, indent=4, ensure_ascii=False)
            
            with open(self.handler_config.data_writing_file, 'r', encoding='utf-8') as f:
                handler_result_list_check = json.load(f)
            assert handler_result_list == handler_result_list_check

            with open(self.handler_config.signal_writing_file, 'w', encoding='utf-8') as f:
                json.dump({'signal': self.handler_config.RESPONSE_SIGNAL}, f, indent=4, ensure_ascii=False)
            print("Function call processing finished")
    
    def post_request(self, server_url: str, query_contents: List[dict],server_cnt):
        if not query_contents:
            return query_contents
        while True:
            try:
                response = requests.post(f"{server_url}/handle_execution", json={"query_contents": query_contents}, timeout=999)
                return response.json()['query_contents']
            except Exception as e:
                print(f"{server_cnt} Error occurred: {e}",flush=True)
                depth += 1
                time.sleep(1)
 

    def handle_single_query(self, query_content: dict, api_result_dict: dict):
        try:
            idx = query_content["idx"]
            question = query_content["question"]
            if idx in self.id_to_context:
                # If question does not match, clear cache
                if len(self.id_to_context[idx]) > 0 and question != self.id_to_context[idx][0].user_query:
                    with self.id_to_context_lock:
                        self.id_to_context[idx] = []
            search_thinking = query_content["think"]
            tool_call = query_content["tool_call"]
            
            assert "name" in tool_call, "no vliad function name in tool_call"
            assert "arguments" in tool_call, "no valid arguments in tool_call"
            assert tool_call["name"] in ["web_search", "browse_webpage"], "invalid tool name"
            if tool_call["name"] == "web_search":
                assert "query" in tool_call["arguments"], "no valid query in tool_call"
                assert isinstance(tool_call["arguments"]["query"], list), "query should be a list"
            elif tool_call["name"] == "browse_webpage":
                assert "url_list" in tool_call["arguments"], "no valid url_list in tool_call"
                assert isinstance(tool_call["arguments"]["url_list"], list), "url_list should be a list"
                assert len(tool_call["arguments"]["url_list"]) >= 1, "url_list number must be greater than 0"

            fuc_name = tool_call["name"]
            arguments = tool_call["arguments"]
            assert fuc_name in ["web_search", "browse_webpage"], "error tool call"
            if fuc_name == "web_search":
                print("web search start!", flush=True) 
                search_query_list = arguments["query"]
                assert isinstance(search_query_list, list), "error search args(query_list)"
                search_query_list = search_query_list[0:3] # only perform first 3 search queries
                web_page_info_list_batch = self.web_search_agent.search_web_batch(user_query=question, search_query_list=search_query_list, api_result_dict=api_result_dict)
                search_result_info_list = [SearchResultInfo(
                    search_query=search_query_list[j],
                    web_page_info_list=web_page_info_list
                )for j, web_page_info_list in enumerate(web_page_info_list_batch)]
                cur_action_info = ActionInfo(user_query=question,
                                                search_thinking=search_thinking,
                                            search_query_list=search_query_list,
                                            search_result_info_list=search_result_info_list)
                if idx not in self.id_to_context:
                    with self.id_to_context_lock:
                        self.id_to_context[idx] = []
                with self.id_to_context_lock:
                    self.id_to_context[idx].append(cur_action_info)
                content = []
                for search_result_info in search_result_info_list:
                    search_query = search_result_info.search_query
                    ret_web_page_info_list = []
                    for web_page_info in search_result_info.web_page_info_list:
                        ret_web_page_info_list.append({
                            "title": web_page_info.title,
                            "url": web_page_info.url,
                            "quick_summary": web_page_info.quick_summary
                        })
                    content.append({
                        "search_query": search_query,
                        "web_page_info_list": ret_web_page_info_list
                    })
                
                return content
            elif fuc_name == "browse_webpage":
                url_list = arguments["url_list"]
                assert isinstance(url_list, list), "error browse args"
                assert len(url_list) >= 1, "browse page number must be greater than 0"
                assert len(self.id_to_context[idx]) >= 1, "no search result"
                assert question == self.id_to_context[idx][-1].user_query, "question not match"
                action_info = self.id_to_context[idx][-1]
                read_webpage_list: List[WebPageInfo] = self.reading_agent.read_batch(user_query=question, search_result_info_list=action_info.search_result_info_list, url_list=url_list, web_search_agent=self.web_search_agent)
                content = []
                for read_webpage in read_webpage_list:
                    information = []
                    for page_read_info in read_webpage.page_read_info_list:
                        if page_read_info.used:
                            continue
                        information.append({
                            "page_number": page_read_info.page_number,
                            "page_summary": page_read_info.page_summary
                        })
                        page_read_info.used = True
                    content.append({
                        "url": read_webpage.url,
                        "information": information
                    })
                return content
            else:
                raise ValueError(f"invalid tool call: {fuc_name}")
        except Exception as e:
            print(f"error: {e}")
            return []
        



if __name__ == "__main__":
    config = yaml.safe_load(open("./scrl/handler/config.yaml"))
    handler_config = SimpleNamespace(
        data_writing_file="./signal/data.json",
        signal_writing_file="./signal/signal.json",
        RESPONSE_SIGNAL=0,
        QUERY_SIGNAL=1,
    )
    client = OpenAI(
        api_key="sk-xxx",
        base_url="xxxx"
    )
    handler = Handler(agent_config=config, client=client, handler_config=handler_config)
    handler.foward_to_server()
    