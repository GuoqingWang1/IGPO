from typing import List, Dict, Any
from web_search_agent.search.text_web_browser import (
    SimpleTextBrowser,
    SearchInformationTool,
    VisitTool,
    PageDownTool
)
from webpage import *

class SubActionInfo:
    def __init__(self, sub_action_thinking, sub_action, take_sub_action_time_taken):
        self.sub_action_thinking = sub_action_thinking
        self.sub_action = sub_action
        self.take_sub_action_time_taken = take_sub_action_time_taken
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sub_action_thinking': self.sub_action_thinking,
            'sub_action': self.sub_action,
            'take_sub_action_time_taken': self.take_sub_action_time_taken
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubActionInfo':
        return cls(
            sub_action_thinking=data['sub_action_thinking'],
            sub_action=data['sub_action'],
            take_sub_action_time_taken=data['take_sub_action_time_taken']
        )

    def __str__(self):
        return str({
            'sub_action_thinking': self.sub_action_thinking,
            'sub_action': self.sub_action,
            'take_sub_action_time_taken': self.take_sub_action_time_taken
        })

class ActionInfo:
    def __init__(self, 
                 user_query,
                 search_thinking, 
                 search_query_list,
                 search_result_info_list: List[SearchResultInfo]):
        self.user_query = user_query
        self.search_thinking = search_thinking
        self.search_query_list: List[str] = search_query_list
        self.search_result_info_list: List[SearchResultInfo] = search_result_info_list
        self.sub_action_info_list: List[SubActionInfo] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_query': self.user_query,
            'search_thinking': self.search_thinking,
            'search_query_list': self.search_query_list,
            'search_result_info_list': [info.to_dict() for info in self.search_result_info_list],
            'sub_action_info_list': [info.to_dict() for info in self.sub_action_info_list],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionInfo':
        instance = cls(
            user_query=data['user_query'],
            search_thinking=data['search_thinking'],
            search_query_list=data['search_query_list'],
            search_result_info_list=[SearchResultInfo.from_dict(info) for info in data['search_result_info_list']],
        )
        
        # Handle sub_action_info_list
        if 'sub_action_info_list' in data:
            instance.sub_action_info_list = [SubActionInfo.from_dict(info) for info in data['sub_action_info_list']]
        
        return instance
        
    def __str__(self):
        return str({
            'user_query': self.user_query,
            'search_thinking': self.search_thinking,
            'search_query_list': self.search_query_list,
            'search_result_info_list': [info.to_dict() for info in self.search_result_info_list],
            'sub_action_info_list': [info.to_dict() for info in self.sub_action_info_list],
        })
        