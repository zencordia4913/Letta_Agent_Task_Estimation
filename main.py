import json
from letta_client import Letta
from letta_client.types import AgentState

AGENT_ID = ""

TEXT_EMBEDDING_LETTA_FREE = "hugging-face/letta-free"
TEXT_EMBEDDING_ADA = "openai/text-embedding-ada-002"
TEXT_EMBEDDING_THREE_SMALL = "openai/text-embedding-3-small"
TEXT_EMBEDDING_THREE_LARGE = "openai/text-embedding-3-large"

HUMAN_MEMORY = {
    "label": "human",
    "value": (
        "This is my section of core memory devoted to information about the human."
        "I don't yet know anything about them."
        "What's their name? Where are they from? What do they do? Who are they?"
        "I should update this memory over time as I interact with the human and learn more about them."
    ),
}
AGENT_MEMORY = {
    "label": "persona",
    "value": (
        "My name is Scalema and I help out BPOSeats clients with business queries."
        "If it's your first time talking with a client be sure to inform them this."
    ),
}


class LettaClient:
    """A LettaClient to make interacting with Letta simple.

    Attributes
    ----------
    client               : Letta
        An instance of the Letta client.
    model                : str
        LLM model to use.
    embedding            : str
        Embedding model to use.
    context_window_limit : int
    """

    model = "openai/gpt-4o"
    embedding = TEXT_EMBEDDING_THREE_SMALL
    context_window_limit = 16000

    def __init__(self, base_url: str):
        """Constructs all necessary attributes for the LettaClient object.

        Parameters
        ----------
        base_url : str
            The base URL to the Letta server.
        """
        self._base_url = base_url
        self.client = Letta(base_url=self._base_url)

        # ! Check to see if provided credentials are valid.
        # ! Raises an error when invalid credentials are provided.
        self.client.agents.list()

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
        agents = self.client.agents.list(**kwargs)
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
        model = kwargs.pop("model", self.model)
        embedding = kwargs.pop("embedding", self.embedding)
        context_window_limit = kwargs.pop(
            "context_window_limit", self.context_window_limit
        )
        message_blocks = [
            kwargs.pop("human_memory", HUMAN_MEMORY),
            kwargs.pop("agent_memory", AGENT_MEMORY),
        ]

        return self.client.agents.create(
            name=name,
            memory_blocks=message_blocks,
            model=model,
            context_window_limit=context_window_limit,
            embedding=embedding,
        )

    def chat_with_agent(self, agent_id: str, message: str, role: str, **kwargs):
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
            response = self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": role, "content": message}],
                **kwargs,
            )

        except ValueError as e:
            error_message = f"Error: {str(e)}"
            if "Agent not found" in str(e):
                error_message = "Invalid agent id provided."

            return None, error_message, True

        messages = []
        resp = ""
        for m in response.messages:
            msg = {}
            match m.message_type:
                case "assistant_message":
                    resp = m.content
                    continue

                case "reasoning_message":
                    msg["content"] = m.reasoning
                    msg["type"] = m.message_type
                    messages.append(msg)
                    continue

                case (
                    "system_message",
                    "tool_call_message",
                    "tool_return_message",
                    "user_message",
                ):
                    msg["content"] = m.content
                    msg["type"] = m.message_type
                    messages.append(msg)
                    continue

                case _:
                    continue

        return resp, messages, False


def main():
    base_url = "http://localhost:8283"
    agent_id = AGENT_ID

    letta = LettaClient(base_url)

    if not agent_id:
        agent = letta.create_agent("Scalema")
        agent_id = agent.id
    else:
        message = "The user is back to chat, start a conversation."
        response, _, error = letta.chat_with_agent(
            agent_id=agent_id, message=message, role="system"
        )
        if error:
            print("An error has occured")
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

        response, response_details, error = letta.chat_with_agent(
            agent_id=agent_id, message=message, role="user"
        )
        if error:
            print(f"System: {response_details}")
            print("*" * 50)
            continue

        print(f"Letta: {response}\n")


if __name__ == "__main__":
    main()
