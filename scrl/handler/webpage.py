from typing import List, Dict, Any
from web_search_agent.search.text_web_browser import SimpleTextBrowser


class WebSelectInfo:
    def __init__(self, web_select_thinking: str, web_select_idx: int):
        self.web_select_thinking: str = web_select_thinking
        self.web_select_idx: int = web_select_idx
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'web_select_thinking': self.web_select_thinking,
            'web_select_idx': self.web_select_idx
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSelectInfo':
        return cls(
            web_select_thinking=data['web_select_thinking'],
            web_select_idx=data['web_select_idx']
        )

class PageReadInfo:
    def __init__(self, 
                 search_results_idx: int, 
                 url: str, 
                 page_title: str,
                 fetch_res: str, 
                 page_thinking: str,
                 page_summary: str,
                 page_number: int, 
                 need_page_down: bool,
                 used: bool=False):
        self.search_results_idx = search_results_idx
        self.url = url
        self.page_title = page_title
        self.fetch_res = fetch_res
        self.page_thinking = page_thinking
        self.page_summary = page_summary
        self.page_number = page_number
        self.need_page_down = need_page_down
        self.used = used
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'search_results_idx': self.search_results_idx,
            'url': self.url,
            'page_title': self.page_title,
            'fetch_res': self.fetch_res,
            'page_thinking': self.page_thinking,
            'page_summary': self.page_summary,
            'page_number': self.page_number,
            'need_page_down': self.need_page_down
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PageReadInfo':
        return cls(
            search_results_idx=data['search_results_idx'],
            url=data['url'],
            page_title=data['page_title'],
            fetch_res=data['fetch_res'],
            page_thinking=data['page_thinking'],
            page_summary=data['page_summary'],
            page_number=data['page_number'],
            need_page_down=data['need_page_down']
        )

    def __str__(self):
        return (f"PageReadInfo(search_results_idx={self.search_results_idx}, "
                f"url='{self.url}', "
                f"page_title='{self.page_title}', "
                f"page_number={self.page_number}, "
                f"need_page_down={self.need_page_down}, "
                f"page_summary='{self.page_summary[:50]}...' if len(self.page_summary) > 50 else '{self.page_summary}')")


class WebPageInfo:
    def __init__(self,
                 title: str,
                 url: str,
                 quick_summary: str,
                 sub_question,
                 browser: SimpleTextBrowser = None):
        self.title = title
        self.url = url
        self.quick_summary = quick_summary
        self.browser = browser
        self.sub_question = sub_question
        self.page_read_info_list: List[PageReadInfo] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'url': self.url,
            'quick_summary': self.quick_summary,
            # Note: browser object might not be serializable directly, 
            # consider adding a separate serialization method if needed
            'sub_question': self.sub_question,
            'page_read_info_list': [info.to_dict() for info in self.page_read_info_list]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], browser=None) -> 'WebPageInfo':
        web_page_info = cls(
            title=data['title'],
            url=data['url'],
            quick_summary=data['quick_summary'],
            browser=browser,  # Browser needs to be passed separately or reconstructed
            sub_question=data['sub_question']
        )
        
        # Reconstruct page_read_info_list
        web_page_info.page_read_info_list = [
            PageReadInfo.from_dict(info_data) 
            for info_data in data.get('page_read_info_list', [])
        ]
        
        return web_page_info
    
    def __str__(self) -> str:
        base_info = f"WebPage: {self.title}\nURL: {self.url}\nQuick Summary: {self.quick_summary}\nSub Question: {self.sub_question}"
        
        if self.page_read_info_list:
            read_info = "\nDetailed Information:"
            for idx, info in enumerate(self.page_read_info_list, 1):
                read_info += f"\n  {idx}. {str(info)}"
            return base_info + read_info
        
        return base_info

class SearchResultInfo:
    def __init__(self, search_query, web_page_info_list: List[WebPageInfo]):
        self.search_query = search_query
        self.web_page_info_list = web_page_info_list
        self.web_select_info_list: List[WebSelectInfo] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'search_query': self.search_query,
            'web_page_info_list': [info.to_dict() for info in self.web_page_info_list],
            'web_select_info_list': [info.to_dict() for info in self.web_select_info_list]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResultInfo':
        instance = cls(
            search_query=data['search_query'],
            web_page_info_list=[WebPageInfo.from_dict(info) for info in data['web_page_info_list']]
        )
        if 'web_select_info_list' in data:
            instance.web_select_info_list = [WebSelectInfo.from_dict(info) for info in data['web_select_info_list']]
        return instance