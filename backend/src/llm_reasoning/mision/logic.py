import openai
import json
from .systems_prompt import SYSTEM_PROMPT

client = openai.Client(api_keys='')

def get_agent_response(user_req):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_req}
        ]
        response_format={"type": "json_object"}
    )
    data = json.loads(response.choices[0].message.content)
    return data["class_ids"]