import os
import re
import openai
from supabase import create_client, Client
import aiohttp
import asyncio
import logging
import warnings
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Literal, Dict, Optional, Any
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from tavily import AsyncTavilyClient
from langgraph.config import get_store
from mcp import McpError
from langchain_mcp_adapters.client import MultiServerMCPClient
from open_deep_research.state import Summary, ResearchComplete
from open_deep_research.configuration import SearchAPI, Configuration
from open_deep_research.prompts import summarize_webpage_prompt, summarize_supabase_prompt
from open_deep_research.logging_config import setup_logging

load_dotenv()

# Get logger for this module
logger = logging.getLogger(__name__)

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args
        queries (List[str]): List of search queries, you can pass in as many queries as you need.
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    # Format the search results and deduplicate results by URL
    formatted_output = f"Search results: \n\n"
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 50_000   # NOTE: This can be tuned by the developer. This character count keeps us safely under input token limits for the latest models.
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop():
        return None
    summarization_tasks = [
        noop() if not result.get("raw_content") else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include],
        )
        for result in unique_results.values()
    ]
    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {
        url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
        for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
    }
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    if summarized_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True, config: RunnableConfig = None):
    tavily_async_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str()))]),
            timeout=180.0
        )
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"""
    except (asyncio.TimeoutError, Exception) as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content

class SupabaseSummary(BaseModel):
    '''Represents a summary of Supabase search results.'''
    summary: str
    key_excerpts: List[str]

async def summarize_supabase_results(model: BaseChatModel, query: str, rag_results: List[Dict]) -> str:
    """Summarize Supabase RAG search results using the summarization model."""
    # Format results for the prompt
    formatted_results = []
    for result in rag_results:
        formatted_results.append({
            "sc_id": result.get('id', ''),
            "section_title": result.get('metadata', {}).get('section_title', ''),
            "doc_title": result.get('metadata', {}).get('doc_title', ''),
            "text": result.get('content', '')
        })
    summary = await asyncio.wait_for(
        model.ainvoke([HumanMessage(content=summarize_supabase_prompt.format(
            query=query,
            rag_results=formatted_results,
            date=get_today_str()
        ))]),
        timeout=180.0
    )

    # Format the response similar to webpage summarization
    excerpts_text = ""
    if hasattr(summary, 'key_excerpts') and summary.key_excerpts:
        for excerpt in summary.key_excerpts:
            excerpts_text += f"[{excerpt.sc_id}] {excerpt.excerpt}\n"

    return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_text}</key_excerpts>"""


##########################
# MCP Utils
##########################
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    try:
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    return None

async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    if current_time > expiration_time:
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    await store.aput((user_id, "tokens"), "data", tokens)
    return

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    old_coroutine = tool.coroutine
    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):
                return exc
            # Handle ExceptionGroup for Python 3.11+
            try:
                if hasattr(exc, 'exceptions'):  # ExceptionGroup-like behavior
                    for sub_exc in exc.exceptions:
                        if found := _find_first_mcp_error_nested(sub_exc):
                            return found
            except AttributeError:
                pass
            return None
        try:
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            mcp_error = _find_first_mcp_error_nested(e_orig)
            if not mcp_error:
                raise e_orig
            error_details = mcp_error.error
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}
            if is_interaction_required:
                message_payload = error_data.get("message", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = (
                        message_payload.get("text") or error_message_text
                    )
                if url := error_data.get("url"):
                    error_message_text = f"{error_message_text} {url}"
                raise ToolException(error_message_text) from e_orig
            raise e_orig
    tool.coroutine = wrapped_mcp_coroutine
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = Configuration.from_runnable_config(config)
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)):
        return []
    tools = []
    # TODO: When the Multi-MCP Server support is merged in OAP, update this code.
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    mcp_server_config = {
        "server_1":{
            "url": server_url,
            "headers": {"Authorization": f"Bearer {mcp_tokens['access_token']}"} if mcp_tokens else None,
            "transport": "streamable_http"
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        mcp_tools = await client.get_tools()
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if tool.name not in set(configurable.mcp_config.tools):
            continue
        tools.append(wrap_mcp_authenticate_tool(tool))
    return tools


##########################
# Tool Utils
##########################
async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        return [search_tool]
    elif search_api == SearchAPI.SUPABASE:
        search_tool = supabase_search
        search_tool.metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    
async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete)]
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# Model Provider Native Websearch Utils
##########################
def anthropic_websearch_called(response):
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        return web_search_requests > 0
    except (AttributeError, TypeError):
        return False

def openai_websearch_called(response):
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if tool_outputs:
        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True
    return False


##########################
# Token Limit Exceeded Utils
##########################
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    return (_check_openai_token_limit(exception, error_str) or
            _check_anthropic_token_limit(exception, error_str) or
            _check_gemini_token_limit(exception, error_str))

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_openai_exception = ('openai' in exception_type.lower() or 
                          'openai' in module_name.lower())
    is_bad_request = class_name in ['BadRequestError', 'InvalidRequestError']
    if is_openai_exception and is_bad_request:
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        if (getattr(exception, 'code', '') == 'context_length_exceeded' or
            getattr(exception, 'type', '') == 'invalid_request_error'):
            return True
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_anthropic_exception = ('anthropic' in exception_type.lower() or 
                             'anthropic' in module_name.lower())
    is_bad_request = class_name == 'BadRequestError'
    if is_anthropic_exception and is_bad_request:
        if 'prompt is too long' in error_str:
            return True
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    is_google_exception = ('google' in exception_type.lower() or 'google' in module_name.lower())
    is_resource_exhausted = class_name in ['ResourceExhausted', 'GoogleGenerativeAIFetchError']
    if is_google_exception and is_resource_exhausted:
        return True
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
}

def get_model_token_limit(model_string):
    for key, token_limit in MODEL_TOKEN_LIMITS.items():
        if key in model_string:
            return token_limit
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]  # Return everything up to (but not including) the last AI message
    return messages

##########################
# Misc Utils
##########################
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    # return datetime.now().strftime("%a %b %-d, %Y")
    return datetime.now().strftime("%Y-%m-%d")

def get_config_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")


SUPABASE_RAG_SEARCH_DESCRIPTION = (
    "A semantic search engine that searches through a curated knowledge base using RAG (Retrieval Augmented Generation). "
    "Useful for finding detailed information from pre-indexed documents and research materials."
)

@tool(description=SUPABASE_RAG_SEARCH_DESCRIPTION)
async def supabase_search(
    queries: List[str],
    match_count: Annotated[int, InjectedToolArg] = 5,
    match_threshold: Annotated[float, InjectedToolArg] = 0.5,
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Supabase RAG search using semantic similarity.

    Args:
        queries (List[str]): List of search queries, you can pass in as many queries as you need.
        match_count (int): Maximum number of results to return per query
        match_threshold (float): Similarity threshold for matching (0-1)

    Returns:
        str: A formatted string of search results
    """
    try:
        from supabase import create_client, Client
        import openai
        
        # Initialize clients
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not supabase_url or not supabase_key:
            return "Error: SUPABASE_URL and SUPABASE_KEY environment variables are required"
        
        if not openai_api_key:
            return "Error: OPENAI_API_KEY environment variable is required for embeddings"
        
        supabase: Client = create_client(supabase_url, supabase_key)
        openai_client = openai.OpenAI(api_key=openai_api_key)
        
    except ImportError:
        return "Error: supabase-py and openai packages are required for Supabase RAG search"
    
    # Get configuration for summarization
    configurable = Configuration.from_runnable_config(config)
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    
    # Collect all results from all queries
    all_results = []
    formatted_output = f"Search results from knowledge base: \n\n"
    
    for query_idx, query in enumerate(queries):
        try:
            # Generate embedding for the query
            embedding_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Call Supabase RPC function
            response = supabase.rpc(
                'match_section_chunks',
                {
                    'query_embedding': query_embedding,
                    'match_count': match_count,
                    'filter': {}  # Empty filter as per RPC signature
                }
            ).execute()
            
            # Filter results by similarity threshold
            filtered_results = [
                doc for doc in response.data 
                if doc.get('similarity', 0) >= match_threshold
            ]
            
            if filtered_results:
                # Add query context to results
                for result in filtered_results:
                    result['query'] = query
                all_results.extend(filtered_results)
                
        except Exception as e:
            formatted_output += f"\n--- QUERY {query_idx + 1}: {query} ---\n"
            formatted_output += f"Error: {str(e)}\n\n"
    
    if not all_results:
        return "No relevant results found in the knowledge base. Please try different search queries or adjust the similarity threshold."
    
    # Group results by query and summarize
    async def noop():
        return None
        
    # Group results by query for summarization
    results_by_query = {}
    for result in all_results:
        query = result['query']
        if query not in results_by_query:
            results_by_query[query] = []
        results_by_query[query].append(result)
    
    # Summarize results for each query
    summarization_tasks = []
    for query, query_results in results_by_query.items():
        if query_results:
            summarization_tasks.append(
                summarize_supabase_results(summarization_model, query, query_results)
            )
        else:
            summarization_tasks.append(noop())
    
    summaries = await asyncio.gather(*summarization_tasks)
    
    # Format the final output
    for i, (query, summary) in enumerate(zip(results_by_query.keys(), summaries)):
        query_results = results_by_query[query]
        formatted_output += f"\n--- QUERY {i + 1}: {query} ---\n"
        formatted_output += f"Found {len(query_results)} relevant results\n\n"
        
        if summary and summary != "":
            formatted_output += f"SUMMARY:\n{summary}\n\n"
        else:
            # Fallback to basic formatting if summarization failed
            for j, doc in enumerate(query_results):
                metadata = doc.get('metadata', {})
                doc_title = metadata.get('doc_title', '')
                section_title = metadata.get('section_title', '')
                sc_id = doc.get('sc_id', '')
                
                if doc_title and section_title:
                    title = f"{doc_title} - Section: {section_title}"
                elif doc_title:
                    title = doc_title
                elif section_title:
                    title = section_title
                else:
                    title = f"Document Chunk {j+1}"
                
                content = doc.get('content', '')
                similarity = doc.get('similarity', 0.0)

                formatted_output += f"SOURCE {j + 1}: {sc_id}\n"
                formatted_output += f"Title: {title}\n"
                formatted_output += f"Relevance: {similarity:.3f}\n"
                formatted_output += f"CONTENT:\n{content[:1000]}{'...' if len(content) > 1000 else ''}\n\n"
        
        formatted_output += "-" * 80 + "\n\n"
    
    return formatted_output

def md_cite_sc_ids(text: str):
    ID_PATTERN     = r'\[?\^?\d+\]?\.?\sID:\s*(sc_[^\s,)]+)'        # capture group 1  ->  sc_xxxxxxx
    PAREN_PATTERN  = r'\(([^)]*ID:\s*sc_[^)]*)\)'  # any (…) that contains at least one ID:
    # PAREN_PATTERN = r'(?<!^\d\.\s)\(([^)]*ID:\s*sc_[^)]*)\)'
    BARE_FOOTNOTE  = r'\[(?!\^)(\d+)\]'
    # 1) collect IDs in first-appearance order ----------------------------
    sc_id_matches = re.findall(ID_PATTERN, text)
    unique_sc_ids = []
    for m in sc_id_matches:
        if m not in unique_sc_ids:
            unique_sc_ids.append(m)
    footnote_map = {sc_id: i + 1 for i, sc_id in enumerate(unique_sc_ids)}

    # 2) replace IDs that live inside parentheses -------------------------
    text = re.sub(PAREN_PATTERN, '', text).strip()

    # 3) replace any remaining standalone IDs -----------------------------
    text = re.sub(ID_PATTERN, '', text).strip()

    # 4) convert bare numeric refs like “[1]” → “[^^1]” -------------------
    text = re.sub(BARE_FOOTNOTE, r'[^\1]', text).strip()

    # 5) append the footnote block ----------------------------------------
    footnotes = [f'[^{num}]: {sc_id}' for sc_id, num in footnote_map.items()]
    text += '\n\n' + '\n'.join(footnotes).strip()
    return text, footnote_map

def _normalize_title(s: str) -> str:
    # Casefold, trim, collapse spaces, drop trailing period
    s = s.strip().rstrip('.')
    s = re.sub(r'\s+', ' ', s)
    return s.casefold()

def dedupe_and_renumber_footnotes(md: str) -> str:
    # Find "### Sources" heading
    HEADING_PATTERN = re.compile(r'(?im)^\s*#{1,6}\s*Sources\s*$', re.M)
    DEF_PATTERN = re.compile(r'(?im)^\s*\[\^(\d+)\]:\s*(.+?)\s*$', re.M)
    REF_PATTERN = re.compile(r'\[\^(\d+)\]')
    h = HEADING_PATTERN.search(md)
    if not h:
        return md  # nothing to do

    # Split body / sources (everything after the heading line is considered sources block)
    heading_line_end = md.find('\n', h.end())
    if heading_line_end == -1:
        body, sources_block = md[:h.start()], ''
    else:
        body, sources_block = md[:h.start()], md[heading_line_end + 1 :]

    # Parse definitions: old_num -> original_title
    defs: Dict[int, str] = {int(n): t.strip() for n, t in DEF_PATTERN.findall(sources_block)}
    if not defs:
        return md  # no definitions found

    # Map normalized title -> first-seen original title (stable representative)
    title_by_norm: Dict[str, str] = {}
    for old_num in sorted(defs):
        t = defs[old_num]
        n = _normalize_title(t)
        if n not in title_by_norm:
            title_by_norm[n] = t

    # Assign new numbers in order of first appearance in the BODY
    norm_to_new: Dict[str, int] = {}
    counter = 1
    for m in REF_PATTERN.finditer(body):
        old = int(m.group(1))
        if old not in defs:
            continue  # unknown reference, leave as-is
        n = _normalize_title(defs[old])
        if n not in norm_to_new:
            norm_to_new[n] = counter
            counter += 1

    # Replace inline references with new indices (only if known & used)
    def _replace_ref(m: re.Match) -> str:
        old = int(m.group(1))
        if old not in defs:
            return m.group(0)  # unknown: keep
        n = _normalize_title(defs[old])
        new_idx = norm_to_new.get(n)
        return f'[^{new_idx}]' if new_idx else m.group(0)

    new_body = REF_PATTERN.sub(_replace_ref, body).rstrip()

    # Rebuild de-duplicated Sources list in new order
    if norm_to_new:
        max_new = max(norm_to_new.values())
        new_defs = []
        # inverse map: index -> normalized title
        idx_to_norm = {v: k for k, v in norm_to_new.items()}
        for i in range(1, max_new + 1):
            n = idx_to_norm.get(i)
            if n is None:
                continue
            title = title_by_norm.get(n, '')
            new_defs.append(f'[^{i}]: {title}')
        new_sources = '### Sources\n\n' + '\n'.join(new_defs) + '\n'
    else:
        # No known refs used in body → keep an empty Sources heading
        new_sources = '### Sources\n'

    return new_body + '\n\n' + new_sources

def format_citations(text: str) -> str:
    """
    Formats citations in the text to use Markdown-style footnotes.
    
    Args:
        text (str): The input text containing citations.
        
    Returns:
        str: The formatted text with citations as footnotes.
    """
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    out, footnote_map = md_cite_sc_ids(text)
    ref_data = supabase.table("ref_info").select("sc_id, doc_title, page_start, page_end").in_("sc_id", list(footnote_map.keys())).execute().data
    logger.info(f"Processing citations for {len(footnote_map)} unique sc_ids: {list(footnote_map.keys())}. Found {len(ref_data)} matching references in Supabase.")
    for each in ref_data:
        # out = out.replace(each['sc_id'], f"{each['doc_title'].title()} p{each['page_start']}{'-'+str(each['page_end']) if each['page_start'] != each['page_end'] else ''}")
        out = out.replace(each['sc_id'], f"{each['doc_title'].title()}")
    out = dedupe_and_renumber_footnotes(out)
    return out
