import json

from letta import EmbeddingConfig, LLMConfig, RESTClient, create_client
from letta.schemas.agent import AgentState
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

AGENT_ID = ""


class LettaClient:
    """A LettaClient to make interacting with Letta simple.

    Attributes
    ----------
    client           : RESTClient
        An instance of the Letta client.
    llm_config       : LLMConfig
        Configuration of LLM model.
    embedding_config : EmbeddingConfig
        Embedding model configuration.
    """

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

    def __init__(self, base_url: str, token: str = None):
        """Constructs all necessary attributes for the LettaClient object.

        Parameters
        ----------
        base_url : str
            The base URL to the Letta server.
        token    : str | None
            The Letta server token.
        """
        self._base_url = base_url

        if token:
            self._headers = {"X-BARE-PASSWORD": f"password {token}"}
            self.client = RESTClient(
                base_url=self._base_url, token=token, headers=self._headers
            )
        else:
            self.client = create_client(base_url=self._base_url)

        # ! Check to see if provided credentials are valid.
        # ! Raises an error when invalid credentials are provided.
        self.client.list_agents()

    def get_agents(self, **kwargs) -> list[dict]:
        """Get a list of available agents.

        Parameters
        ----------
        **kwargs : any
            Any extra parameters when listing agents.

        Returns
        -------
        list[dict]
            Returns a list of dicts containing ``id`` and ``name`` keys.
        """
        agents = self.client.list_agents(**kwargs)
        return [{"id": agent.id, "name": agent.name} for agent in agents]

    def create_agent(self, name: str, **kwargs) -> AgentState:
        """Create a new Letta agent.

        Parameters
        ----------
        name : str
            The name of the agent to be created.
        **kwargs : any
            Extra details needed to create an agent.

        Returns
        -------
        AgentState
            The created agent instance.
        """
        llm_config = kwargs.pop("llm_config", self.llm_config)
        embedding_config = kwargs.pop("embedding_config", self.embedding_config)

        return self.client.create_agent(
            name=name,
            llm_config=llm_config,
            embedding_config=embedding_config,
            **kwargs,
        )

    def chat_with_agent(
        self, agent_id: str, message: str, role: str, **kwargs
    ) -> tuple[str, bool]:
        """Send a message to an agent and get the response.

        Parameters
        ----------
        agent_id : str
            The agent ID to send the message to.
        message  : str
            The message to send to the agent.
        role     : str
            The role of the person sending the message that is, user or system.
        **kwargs : any
            Extra parameters needed to send a message.

        Returns
        -------
        str
            The bot response.
        bool
            True if sending the message was unsuccessful. False otherwise.

        Raises
        ------
        ValueError :
            Raises a value error when invalid role string is provided

        """
        if role not in ["user", "system"]:
            raise ValueError("Role must be either 'user' or 'system'")

        try:
            response = self.client.send_message(
                message=message,
                role=role,
                agent_id=agent_id,
                **kwargs,
            )

        except ValueError as e:
            error_message = "An error has occurred."
            if "Agent not found" in str(e):
                error_message = "Invalid agent id provided."

            return error_message, True

        bot_response = (json.loads(response.messages[1].tool_call.arguments))["message"]
        return bot_response, False


def main():
    agent_id = AGENT_ID
    letta = LettaClient("http://localhost:8283")

    if not agent_id:
        agent_info = {
            "memory": ChatMemory(persona=AGENT_PERSONA, human=AGENT_HUMAN_INFO),
        }
        agent = letta.create_agent("Scalema", **agent_info)
        agent_id = agent.id
    else:
        message = "The user is back to chat, start a conversation."
        response, error = letta.chat_with_agent(
            agent_id=agent_id, message=message, role="system"
        )
        if error:
            print("An error has occurred")
            return

        print(f"Letta: {response}\n")

    while True:
        message = input("> ")
        message = message.strip()
        if not message:
            print("System: Please enter a valid string. \n")
            print("*" * 50)
            continue

        if message == ".exit":
            break

        response, error = letta.chat_with_agent(
            agent_id=agent_id, message=message, role="user"
        )
        if error:
            print(f"System: {response}")
            print("*" * 50)
            continue

        print(f"Letta: {response}\n")


if __name__ == "__main__":
    main()
