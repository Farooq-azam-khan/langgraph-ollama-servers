from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated


async def search_wiki(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search form wiki."""
    wiki_tool = WikipediaQueryRun(
        name="wiki-query-tool", api_wrapper=WikipediaAPIWrapper()
    )
    result = await wiki_tool.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [search_wiki]
