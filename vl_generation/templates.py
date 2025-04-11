### ------------ input templates -------------
ALL_INFO_INPUT_TEMPLATE = '''
Scene annotations:
{{
    "object information": {}.
}}
Motion annotations:
{{
    "action": {}.
    "key_moments": {}.
}}
'''

ACTION_INPUT_TEMPLATE = '''
{{
    "action": {}.
}}
'''

PRED_INPUT_TEMPLATE = '''
Scene annotations:
{{
    "object information": {}
}}
Motion annotations:
{{
    "action": {}
    "current key moments": {}
    "future key moments": {}
}}
'''

### ------------- output templates ------------
QA_OUTPUT_TEMPLATE = '''
Question: {}
Answer: {}
'''

DIALOGUE_OUTPUT_TEMPLATE = '''
Human: {}
Agent: {}
'''