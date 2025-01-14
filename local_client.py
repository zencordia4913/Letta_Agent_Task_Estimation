import json

from letta import EmbeddingConfig, LLMConfig, create_client
from letta.schemas.memory import ChatMemory

AGENT_HUMAN_INFO = """
    This is my section of core memory devoted to information about the human.
    I don't yet know anything about them.
    What's their name? Where are they from? What do they do? Who are they?
    I should update this memory over time as I interact with the human and learn more about them.
"""

AGENT_PERSONA = """
My name is Scalema and I help out BPOSeats clients with business queries.
"""


class LettaClient:
    llm_config = LLMConfig(
        model="gpt-4o",
        model_endpoint="https://api.openai.com/v1",
        model_endpoint_type="openai",
        context_window=128000,
    )
    embedding_config = EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        embedding_model="text-embedding-ada-002",
        embedding_dim=1536,
        embedding_chunk_size=300,
    )

    def __init__(self, base_url: str):
        self._base_url = base_url
        self.client = create_client(self._base_url)
        self.client.list_agents()

    def get_agents(self):
        agents = self.client.list_agents()
        return [{"id": agent.id, "name": agent.name} for agent in agents]

    def create_agent(self, name: str, **kwargs):
        llm_config = kwargs.pop("llm_config", self.llm_config)
        embedding_config = kwargs.pop("embedding_config", self.embedding_config)

        return self.client.create_agent(
            name=name,
            llm_config=llm_config,
            embedding_config=embedding_config,
            **kwargs,
        )

    def chat_with_agent(self, agent_id: str, message: str, **kwargs):
        try:
            response = self.client.send_message(
                message=message,
                role="user",
                agent_id=agent_id,
                **kwargs,
            )

        except ValueError:
            return "", True

        bot_response = (json.loads(response.messages[1].tool_call.arguments))["message"]
        return bot_response, False

    def system_message(self, agent_id, message):
        self.client.send_message(
            message=message,
            role="system",
            agent_id=agent_id,
        )


if __name__ == "__main__":
    letta = LettaClient("http://localhost:8283")

    # agent_info = {
    #     "memory": ChatMemory(persona=AGENT_PERSONA, human=AGENT_HUMAN_INFO),
    # }
    # agent = letta.create_agent("Scalema", **agent_info)
    # print(agent.id)
    agent_id = "agent-0bf0f16d-02e5-4daf-b8bf-582cdee42b3d"
    message = "The user has opened the tab, please initiate a conversation."
    letta.system_message(agent_id, message)
