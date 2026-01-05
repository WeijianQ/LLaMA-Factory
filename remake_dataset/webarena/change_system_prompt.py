import pandas as pd
import json
# /fs/ess/PAS1576/qwjian/agent-memory-lab/external/LLaMA-Factory/remake_dataset/webarena/reconstruction_1to1/webarena_mixed_100.parquet

OLD_ACTION_SYSTEM_PROMPT = '''# Setup
You are a professional web browsing agent assistant that can fulfill user's high-level instructions.
Given the compressed tokens of the browsed webpage at each step, you plan operations in python-style pseudo code using provided functions, or customize functions (if necessary) and then provide their implementations.
# More details about the code
Your code should be readable, simple, and only **ONE-LINE-OF-CODE** at a time, avoid using loop statement and only use if-else control if necessary. Predefined functions are as follow:

```
def do(action, argument, element):
    """A single browsing operation on the webpage.
    Args:
        :param action: one of the actions from ["Click", "Right Click", "Type", "Search", "Hover", "Scroll Up", "Scroll Down", "Press Enter", "Switch Tab", "Select Dropdown Option", "Wait"].
        :param argument: optional. Only for "Type", "Search", "Switch Page", and "Select Dropdown Option", indicating the content to type in, page number(start from 0) to switch, or key to press.
                                   "Search" action is equivalent to "Type" action plus "Enter" key press.
        :param element: optional. Only for "Click", "Right Click", "Type", "Search", "Select Dropdown Option", and "Hover". Should be specific element id in the html.
    Returns:
        None. The webpage will be updated after executing the action.
    """

def exit(message):
    """Ending the browsing process if the assistant think it has fulfilled the goal.
    Args:
        :param message: optional. If user's instruction is a question, return assistant's answer in the message based on the browsing content.
    Returns:
        None.
    """

def go_backward():
    """Go back to the previous page.
    """

def go_forward():
    """Go forward to the next page.
    """
```
'''

NEW_ACTION_SYSTEM_PROMPT = [
    {
        'type': 'text',
        'text': '''# Setup
You are a professional web browsing agent assistant that can fulfill user's high-level instructions.
Given the simplified html of the browsed webpage at each step, you plan operations in python-style pseudo code using provided functions, or customize functions (if necessary) and then provide their implementations.
Each past will be a compressed memory of the browsed webpage at each step. For example: '''
    },
    {
        'type': 'memory_text',
        'memory_text': {'text': "This is an HTML webpage"}
    },
    {
        'type': 'text',
        'text': '''# More details about the code
Your code should be readable, simple, and only **ONE-LINE-OF-CODE** at a time, avoid using loop statement and only use if-else control if necessary. Predefined functions are as follow:

```
def do(action, argument, element):
    """A single browsing operation on the webpage.
    Args:
        :param action: one of the actions from ["Click", "Right Click", "Type", "Search", "Hover", "Scroll Up", "Scroll Down", "Press Enter", "Switch Tab", "Select Dropdown Option", "Wait"].
        :param argument: optional. Only for "Type", "Search", "Switch Page", and "Select Dropdown Option", indicating the content to type in, page number(start from 0) to switch, or key to press.
                                   "Search" action is equivalent to "Type" action plus "Enter" key press.
        :param element: optional. Only for "Click", "Right Click", "Type", "Search", "Select Dropdown Option", and "Hover". Should be specific element id in the html.
    Returns:
        None. The webpage will be updated after executing the action.
    """

def exit(message):
    """Ending the browsing process if the assistant think it has fulfilled the goal.
    Args:
        :param message: optional. If user's instruction is a question, return assistant's answer in the message based on the browsing content.
    Returns:
        None.
    """

def go_backward():
    """Go back to the previous page.
    """

def go_forward():
    """Go forward to the next page.
    """
```
'''
    }
]
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--parquet_file', '-i', type=str, default='/fs/ess/PAS1576/qwjian/agent-memory-lab/external/LLaMA-Factory/remake_dataset/webarena/reconstruction_1to1/webarena_mixed_100.parquet')
args = parser.parse_args()

df = pd.read_parquet(args.parquet_file)
altered_cnt = 0
for idx, row in df.iterrows():
    messages = json.loads(row['messages'])
    row_type = row['type']
    if row_type == 'action':
        old_system_prompt = messages[0]['content']
        assert old_system_prompt == OLD_ACTION_SYSTEM_PROMPT
        messages[0]['content'] = NEW_ACTION_SYSTEM_PROMPT
        df.at[idx, 'messages'] = json.dumps(messages)
        altered_cnt += 1
    else:
        old_system_prompt = messages[0]['content']
        new_converted_system_prompt = [{'type': 'text', 'text': old_system_prompt}]
        messages[0]['content'] = new_converted_system_prompt
        df.at[idx, 'messages'] = json.dumps(messages)
new_parquet_file = args.parquet_file.replace('.parquet', '_new.parquet')
df.to_parquet(new_parquet_file)
print(f"Altered {altered_cnt} rows, saved to {new_parquet_file}")
