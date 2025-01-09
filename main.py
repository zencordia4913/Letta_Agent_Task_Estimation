import json
from http import HTTPStatus

import requests

AGENT_ID = ""


def get_letta_response(agent_id: str, message: str):
    url = f"http://localhost:8283/v1/agents/{agent_id}/messages"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "user", "text": message},
        ],
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code != HTTPStatus.OK:
        return "", True

    bot_response = ""
    response_data = response.json()
    for m in response_data["messages"]:
        if "tool_call" in m:
            tool_call_arguments = m["tool_call"]["arguments"]
            bot_response = (json.loads(tool_call_arguments))["message"]
            break

    return bot_response, False


if __name__ == "__main__":
    i = 0
    while i < 1:
        message = input("> ")
        if message == ".exit":
            i += 1
            continue

        resp, error = get_letta_response(agent_id=AGENT_ID, message=message)
        if error:
            i += 1
            continue

        print(f"Letta: {resp}\n")
