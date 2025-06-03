import logging
import os
import sys
import json
from letta_client import Letta
from letta_client.types import AgentState

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"OPENAI_API_KEY is detected: {openai_api_key[:5]}********") 
else:
    print("OPENAI_API_KEY is NOT detected. Check your .env file.")


os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    level=logging.DEBUG,  # Log everything (DEBUG and above)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/debug.log", mode="w"),  # Save logs to file
        logging.StreamHandler(sys.stdout)  # Print logs to console
    ]
)

logging.info("Logging initialized successfully.")

AGENT_ID = ""

TEXT_EMBEDDING_LETTA_FREE = "hugging-face/letta-free"
TEXT_EMBEDDING_ADA = "openai/text-embedding-ada-002"
TEXT_EMBEDDING_THREE_SMALL = "openai/text-embedding-3-small"
TEXT_EMBEDDING_THREE_LARGE = "openai/text-embedding-3-large"

HUMAN_MEMORY = {
    "label": "human",
    "value": "This is my section of core memory devoted to information about the human."
}

AGENT_MEMORY = {
    "label": "persona",
    "value": "My name is Scalema and I help out BPOSeats clients with business queries."
}



class LettaClient:
    """Handles interaction with the Letta API."""

    model = "letta/letta-free"
    embedding = TEXT_EMBEDDING_THREE_SMALL
    context_window_limit = 8192

    def __init__(self, base_url: str):
        logging.info("Initializing Letta Client...")
        self._base_url = base_url
        self.client = Letta(base_url=self._base_url)

        try:
            self.client.agents.list()
            # List available LLMs
            models = self.client.models.list_llms()
            print("Available LLMs:", models)
            logging.info("Successfully connected to Letta API.")
            embedding_models = self.client.models.list_embedding_models()
            print("Available Embedding Models:", embedding_models)
        except Exception as e:
            logging.error(f"Failed to connect to Letta API: {e}")
            raise

    def create_agent(self, name: str, **kwargs) -> AgentState:
        """Create a new Letta agent and attach the tool, with full error logging."""
        logging.info(f"📌 Creating new agent: {name}")

        model="letta/letta-free"  
        embedding = "letta/letta-free"
        context_window_limit = kwargs.pop("context_window_limit", self.context_window_limit)
        message_blocks = [HUMAN_MEMORY, AGENT_MEMORY]

        logging.info("Registering tool before creating agent...")
        tool = self.register_tool()

        try:
            agent = self.client.agents.create(
                name=name,
                memory_blocks=message_blocks,
                model=model,
                context_window_limit=context_window_limit,
                embedding=embedding,
                tool_ids=[tool.id]
            )
            logging.info(f"Agent created successfully! ID: {agent.id}, Model: {model}")
            return agent

        except Exception as e:
            logging.error(f"Error while creating agent: {e}")

            
            if hasattr(e, "status_code") and hasattr(e, "body"):
                logging.error(f"🔍 API Response: {e.status_code}, Body: {e.body}")
            
            raise  

    def register_tool(self):
        """Check if `predict_task_duration` is already registered before creating it."""
        logging.info("Checking if `predict_task_duration` is already registered...")


        tools = self.client.tools.list()
        for tool in tools:
            if tool.name == "extract_task_duration":
                logging.info(f"Tool `{tool.name}` already exists. Using existing tool ID: {tool.id}")
                return tool  

        # If not found, register the tool
        logging.info("Tool not found. Registering `predict_task_duration`...")
        tool = self.client.tools.create_from_function(func=extract_task_duration)
        logging.info(f"Registered tool: {tool.name} (ID: {tool.id})")

        return tool

    def chat_with_agent(self, agent_id: str, message: str, role: str, **kwargs):
        """Send a message to an agent and get the response."""
        logging.info(f"Sending message to agent {agent_id}: {message}")

        if role not in ["user", "system"]:
            raise ValueError("Role must be either 'user' or 'system'")

        try:
            response = self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": role, "content": message}],
                **kwargs,
            )
            logging.info(f"Received response from agent {agent_id}")
        except ValueError as e:
            logging.error(f"Error in chat_with_agent: {str(e)}")
            return None, str(e), True


        messages = []
        assistant_response = ""

        for m in response.messages:
            msg = {}

            if hasattr(m, "content"):
                assistant_response = m.content  # Normal case: assistant response
            elif hasattr(m, "reasoning"):
                msg["content"] = m.reasoning
                msg["type"] = "reasoning_message"
                messages.append(msg)
                continue
            elif hasattr(m, "message_type"):
                msg["content"] = getattr(m, "content", "No content")
                msg["type"] = m.message_type
                messages.append(msg)
                continue

        return assistant_response, messages, False




# def predict_task_duration(task_name: str) -> str:
#     """
#     Estimates the number of hours required to complete a given task.

#     Parameters:
#         task_name (str): A string representing the name or description of the task 
#                          that needs to be estimated.

#     Returns:
#         str: A string containing the estimated duration in hours.
#     """
    
#     import requests

#     DGX_SERVER_URL = "http://202.92.159.242:8001/predict/"  
#     payload = {"task_name": task_name}
    
#     try:
#         response = requests.post(DGX_SERVER_URL, json=payload, timeout=10)  
        
#         if response.status_code == 200:
#             result = response.json()
#             estimated_duration = result.get("estimated_duration", "N/A")
#             inference_time = result.get("inference_time", "N/A")
#             memory_used = result.get("memory_used", "N/A")

#             return (f"Estimated Task Duration: {estimated_duration:.2f} hours\n"
#                     f"Inference Time: {inference_time} ms\n"
#                     f"Memory Used: {memory_used} MB")

#         else:
#             return f"Error: Received {response.status_code} from DGX server: {response.text}"

#     except requests.RequestException as e:
#         return f"Error: Failed to connect to DGX server: {str(e)}"
    

def extract_task_duration(task_name: str) -> str:
    """
    Estimates the number of hours required to complete a given task.

    Parameters:
        task_name (str): A string representing the name or description of the task 
                         that needs to be estimated.

    Returns:
        str: A string containing the estimated duration in hours.
    """
    
    import requests
    import psycopg2
    import logging
    DGX_API_URL_PROD = "http://202.92.159.242:8001/prod_infer"
    logging.debug("CONTACTING THE DGX SERVER")

    try:
        response = requests.post(
                DGX_API_URL_PROD, 
                json={"task_name": task_name}
        )
        response.raise_for_status()
        
        embedding = response.json()
        conn = psycopg2.connect(
                dbname="omni_data",
                user="jeryl4913",
                password="bposeats", 
                host="host.docker.internal", 
                port="5433"
        )
        cur = conn.cursor()
        embedding_str = "[" + ",".join(str(x) for x in embedding[0]) + "]"

        sql = """
                SELECT title, task_duration_hours
                FROM task_embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT 1;
        """
        cur.execute(sql, (embedding_str,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        return str(result[1])

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None




def main():
    logging.info("Starting main script.")
    base_url = "http://localhost:8283"
    agent_id = AGENT_ID

    letta = LettaClient(base_url)

    if not agent_id:
        agent = letta.create_agent("Scalema")
        agent_id = agent.id

    while True:
        message = input("> ").strip()
        if not message:
            print("System: Please enter a valid message.")
            continue
        if message == ".exit":
            break

        response, _, error = letta.chat_with_agent(agent_id=agent_id, message=message, role="user")
        if error:
            print("Error:", response)
            continue

        print(f"Letta: {response}\n")


if __name__ == "__main__":
    main()
