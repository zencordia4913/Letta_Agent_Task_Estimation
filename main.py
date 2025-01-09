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


def is_agent_id_valid(agent_id: str) -> bool:
    url = f"http://localhost:8283/v1/agents/{agent_id}/"
    response = requests.get(url)

    return response.status_code == HTTPStatus.OK


if __name__ == "__main__":
    if not AGENT_ID:
        raise ValueError("AGENT_ID can not be empty")

    if not is_agent_id_valid(AGENT_ID):
        raise ValueError("Please set a valid AGENT_ID")

    while True:
        message = input("> ")
        message = message.strip()
        if not message:
            print("System: Please enter a valid string. \n")
            print("*"*100)
            continue

        if message == ".exit":
            break

        resp, error = get_letta_response(agent_id=AGENT_ID, message=message)
        if error:
            break

        print(f"Letta: {resp}\n")
