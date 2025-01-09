# Letta Terminal Example

A simple terminal Letta implementation. This is a proof of concept on how to work with Letta

### Requirements

- Python-3.10

- Docker

- Make (optional)

## Getting started

- Clone the repo

- Create and activate a virtual environment

```sh
python3.10 -m venv venv

source venv/bin/activate
```

- Install python dependencies

```sh
pip install -r requirements.txt
```

- Create a `.env` and add `OPEN_API_KEY`. Use your local GPT key

```sh
touch .env

echo "OPEN_API_KEY='<LOCAL_GPT_KEY>'" >> .env
```

- Start the Letta docker service

```sh
sudo make up

## If you don't have make installed, use the following
sudo docker compose -f docker-compose.yaml up
```

- Set up a Letta agent. Got to [https://app.letta.com/](https://app.letta.com/) and login with `Google`.

- Create a new agent. Preferably `Internet chatbot`.

- Copy the agent ID below the agent name and paste it in the `main.py`

- In another terminal session but the same directory, start the terminal app.

```sh
python main.py
```

- Start chatting with the bot.

### How to stop the chat

- Type `.exit`

### How to stop the Letta server

- Type `Ctrl + C`

- Take down the docker instance

```sh
sudo make down

## If you don't have make installed, use the following
sudo docker compose -f docker-compose.yaml down
```
