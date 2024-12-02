from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_tool = WikipediaQueryRun(name="wiki-query-tool", api_wrapper=WikipediaAPIWrapper())
