from langchain_ollama import ChatOllama
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from wa_tool import wiki_tool
from langgraph.prebuilt import ToolNode, tools_condition

llm = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.7, num_predict=100)
llm_with_wiki = llm.bind_tools([wiki_tool])


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm_with_wiki.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[wiki_tool]))
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


with open("./compiled/chatbot.png", "wb") as f:
    png = graph.get_graph().draw_mermaid_png()
    f.write(png)


def main():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except Exception as e:
            raise e


if __name__ == "__main__":
    pass
