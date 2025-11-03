import yaml
from time import strftime, gmtime
from jinja2 import StrictUndefined, populate_template
from typing import Dict, Any

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        return Exception("Error!")

def initializa_system_prompt(system_prompt_templates, tools, today) -> str:
    system_prompt = populate_template(
        system_prompt_templates,
        variables={
            "tools": tools,
            "today": today
        },
    )
    return system_prompt