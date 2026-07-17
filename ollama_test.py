# from ollama import Client
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = Client(
#     host=os.environ["OLLAMA_BASE_URL"],
#     headers={
#         "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
#     }
# )

# response = client.chat(
#     model="gpt-oss:20b",   # or whichever cloud model you have access to
#     messages=[
#         {
#             "role": "user",
#             "content": "Say hello."
#         }
#     ]
# )

# print(response["message"]["content"])



import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOllama(
    model="gpt-oss:20b",
    base_url=os.environ["OLLAMA_BASE_URL"],
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
        }
    },
)

response = llm.invoke(
    [HumanMessage(content="Reply with exactly: ChatOllama works")]
)

print(response.content)