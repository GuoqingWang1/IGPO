from typing import List, Dict, Any
from utils import (
    get_content_from_tag,
    get_response_from_llm
)
from .prompts import *
from webpage import *
import time
import random
import html2text
import concurrent.futures

class ReadingAgent:
    def __init__(self,
                 config,
                 client):
        self.config = config
        self.client = client
        
    def read(
            self,
            main_question,
            sub_question,
            selected_result_idx: int,
            cur_webpage: WebPageInfo,
            context: List[WebPageInfo] = [],
            web_search_agent = None
    ):
        if cur_webpage.browser == "error":
            return cur_webpage
        if cur_webpage.browser is None:
            cur_webpage.browser = web_search_agent.scrape_and_check_valid_api(cur_webpage.url)
            if cur_webpage.browser is None:
                cur_webpage.browser = "error"
                return cur_webpage
        context_so_far_prefix = ""
        for webpage in context:
            useful_info = ""
            for page_read_info in webpage.page_read_info_list:
                useful_info += page_read_info.page_summary + "\n\n"
            if len(useful_info):
                context_so_far_prefix += f"<sub_question>{webpage.sub_question}</sub_question>\n<useful_info>{useful_info}</useful_info>\n"

        cur_useful_info = ""
        total_pages = len(cur_webpage.browser.viewport_pages)
        while cur_webpage.browser.viewport_current_page < total_pages:
            context_so_far = ""
            if cur_useful_info:
                context_so_far = context_so_far_prefix + f"<sub_question>{sub_question}</sub_question>\n<useful_info>{cur_useful_info}</useful_info>"
            else:
                context_so_far = context_so_far_prefix
            cur_web_page_content = cur_webpage.browser._state()[1]
            cur_web_page_content = html2text.html2text(cur_web_page_content)
            page_index = cur_webpage.browser.viewport_current_page + 1
            prompt = EXTRACT_NEW_INFO_PROMPT.format(
                main_question=main_question,
                sub_question=sub_question,
                context_so_far=context_so_far.strip(),
                page_index=page_index,
                total_pages=total_pages,
                page_content=cur_web_page_content
            )

            messages = [{"role": "user", "content": prompt}]
            response = get_response_from_llm(
                messages=messages,
                client=self.client,
                model=self.config["reading_agent_model"],
                stream=False
            )
            
            extracted_info = get_content_from_tag(response["content"], "extracted_info", "").strip()
            page_down = get_content_from_tag(response["content"], "page_down", "").strip()
            short_summary = get_content_from_tag(response["content"], "short_summary", "").strip()

            if "yes" in page_down:
                page_down = True
            else:
                page_down = False

            if extracted_info:
                cur_webpage.page_read_info_list.append(
                    PageReadInfo(
                        search_results_idx=selected_result_idx,
                        url=cur_webpage.url,
                        page_title=cur_webpage.title,
                        fetch_res=cur_web_page_content,
                        page_thinking=response["reasoning_content"] if "reasoning_content" in response else "",
                        page_summary=extracted_info,
                        page_number=cur_webpage.browser.viewport_current_page,
                        need_page_down=page_down,
                        used=False,
                    )
                )
                cur_useful_info += extracted_info + "\n\n"

            if page_down:
                cur_webpage.browser.page_down()
            else:
                break
        return cur_webpage

    def read_batch(
            self,
            user_query: str,
            search_result_info_list: List[SearchResultInfo],
            url_list: List[str],
            web_search_agent = None,
    ):
        
        url_dict = {}
        for url in url_list:
            url_dict[url] = []
        future_to_content = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            for search_result_info in search_result_info_list:
                search_query = search_result_info.search_query
                web_page_info_list = search_result_info.web_page_info_list
                for selected_result_idx, cur_webpage in enumerate(web_page_info_list):
                    if cur_webpage.url not in url_dict:
                        continue
                    future = executor.submit(self.read,
                                            user_query,
                                            search_query,
                                            selected_result_idx,
                                            cur_webpage,
                                            web_page_info_list,
                                            web_search_agent)
                    future_to_content.append(future)
        read_webpage_list = []
        for i, future in enumerate(future_to_content):
            cur_webpage: WebPageInfo = future.result()
            read_webpage_list.append(cur_webpage)
        return read_webpage_list
