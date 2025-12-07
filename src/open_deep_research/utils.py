"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import logging
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt, summarize_supabase_prompt
from open_deep_research.state import ResearchComplete, Summary, SupabaseSummary

##########################
# Model Initialization Utils
##########################

def init_chat_model_with_gpt5_support(
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    tags: Optional[List[str]] = None,
    configurable_fields: Optional[tuple] = None,
    **kwargs
) -> BaseChatModel:
    """Initialize a chat model with special handling for GPT-5 models.
    
    GPT-5 models require the Responses API (/v1/responses) instead of 
    Chat Completions API (/v1/chat/completions). LangChain's recent versions
    of langchain-openai (0.3.0+) automatically detect GPT-5 models and use
    the correct endpoint.
    
    If you're getting a 404 error with GPT-5 models, ensure you have the latest
    version of langchain-openai:
        pip install --upgrade langchain-openai
    
    Args:
        model: Model identifier string (e.g., "openai:gpt-5", "gpt-5-nano")
        max_tokens: Maximum tokens for completion
        api_key: API key for authentication
        tags: Optional tags for tracing
        configurable_fields: Fields that can be configured at runtime
        **kwargs: Additional arguments passed to init_chat_model
        
    Returns:
        Initialized chat model instance
    """
    # Check if this is a GPT-5 model (if model is provided)
    if model:
        model_lower = model.lower()
        is_gpt5 = 'gpt-5' in model_lower or 'gpt5' in model_lower
        
        if is_gpt5:
            logging.info(f"Initializing GPT-5 model: {model}")
            logging.info("GPT-5 models use the Responses API (/v1/responses) endpoint")
            logging.info("Ensure langchain-openai >= 0.3.0 for GPT-5 support")
    
    # Initialize the model using standard init_chat_model
    # LangChain's recent versions should automatically detect GPT-5 and use responses API
    init_kwargs = {
        "max_tokens": max_tokens,
        "api_key": api_key,
        "tags": tags,
        "configurable_fields": configurable_fields,
        **kwargs
    }
    
    # Remove None values
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    
    # Call with model if provided, otherwise without it (for configurable models)
    try:
        if model:
            return init_chat_model(model=model, **init_kwargs)
        else:
            return init_chat_model(**init_kwargs)
    except Exception as e:
        # Check if this is the GPT-5 endpoint error
        error_msg = str(e)
        if "404" in error_msg and "v1/responses" in error_msg:
            raise RuntimeError(
                f"GPT-5 model initialization failed. "
                f"Please upgrade langchain-openai: pip install --upgrade langchain-openai"
            ) from e
        raise

##########################
# Logging Configuration
##########################

def setup_logging():
    """Configure logging to write to a timestamped file in the logs directory."""
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"deep_research_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger to catch all logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Also add to this module's logger
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.INFO)
    
    logging.info(f"Logging initialized - writing to {log_file}")
    return log_file

# Initialize logging when module is imported
_log_file_path = setup_logging()

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
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # Step 3: Set up the summarization model with configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Character limit to stay within model token limits (configurable)
    max_char_to_include = configurable.max_content_length
    
    # Initialize summarization model with retry logic
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model_with_gpt5_support(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # Step 4: Create summarization tasks (skip empty content)
    async def noop():
        """No-op function for results without raw content."""
        return None
    
    summarization_tasks = [
        noop() if not result.get("raw_content") 
        else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]
    
    # Step 5: Execute all summarization tasks in parallel
    summaries = await asyncio.gather(*summarization_tasks)
    
    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # Step 7: Format the final output
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.
    
    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access
        
    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content using AI model with timeout protection.
    
    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized
        
    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    try:
        # Create prompt with current date context
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=120.0  # 120 second timeout for summarization
        )
        
        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        # Other errors during summarization - log and return original content
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

async def summarize_supabase_results(model: BaseChatModel, query: str, results: list) -> str:
    """Summarize Supabase RAG search results using the summarization model.
    
    Args:
        model: The chat model configured for summarization
        query: The search query that produced these results
        results: List of search result dictionaries from Supabase
        
    Returns:
        Formatted summary with key excerpts tagged by sc_id
    """
    # Format results for the prompt
    formatted_results = []
    for result in results:
        formatted_results.append({
            "sc_id": result.get('id', ''),
            "section_title": result.get('metadata', {}).get('section_title', ''),
            "doc_title": result.get('metadata', {}).get('doc_title', ''),
            "text": result.get('content', '')
        })
    
    try:
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=summarize_supabase_prompt.format(
                query=query,
                rag_results=formatted_results,
                date=get_today_str()
            ))]),
            timeout=600.0
        )

        # Format the response with excerpts tagged by sc_id
        excerpts_text = ""
        if hasattr(summary, 'key_excerpts') and summary.key_excerpts:
            for excerpt in summary.key_excerpts:
                excerpts_text += f"[{excerpt.sc_id}] {excerpt.excerpt}\n"

        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_text}</key_excerpts>"""
    
    except asyncio.TimeoutError:
        # Timeout during summarization - return basic formatted content
        logging.warning("Supabase results summarization timed out after 600 seconds, returning basic formatting")
        return ""
    except Exception as e:
        # Other errors during summarization - log and return empty string to trigger fallback
        logging.warning(f"Supabase results summarization failed with error: {str(e)}, using fallback formatting")
        return ""

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP Utils
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If MCP server connection fails, return empty list
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools

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
            error_msg = "Error: SUPABASE_URL and SUPABASE_KEY environment variables are required but not set. Please configure these in your .env file."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if not openai_api_key:
            error_msg = "Error: OPENAI_API_KEY environment variable is required for embeddings but not set. Please configure this in your .env file."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        supabase: Client = create_client(supabase_url, supabase_key)
        openai_client = openai.OpenAI(api_key=openai_api_key)
        
    except ImportError as e:
        error_msg = f"Error: supabase-py and openai packages are required for Supabase RAG search. Install with: pip install supabase openai. Details: {e}"
        logging.error(error_msg)
        raise ImportError(error_msg)
    
    logging.info(f"Supabase search executing with {len(queries)} queries")
    
    # Get configuration for summarization
    configurable = Configuration.from_runnable_config(config)
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model_with_gpt5_support(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(SupabaseSummary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    
    # Collect all results from all queries
    all_results = []
    formatted_output = f"Search results from knowledge base: \n\n"
    
    for query_idx, query in enumerate(queries):
        try:
            logging.info(f"Executing Supabase search for query {query_idx + 1}/{len(queries)}: {query}")
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
            
            logging.info(f"Query {query_idx + 1} returned {len(filtered_results)} results above threshold {match_threshold}")
            
            if filtered_results:
                # Add query context to results
                for result in filtered_results:
                    result['query'] = query
                all_results.extend(filtered_results)
                
        except Exception as e:
            error_msg = f"Error executing Supabase search for query '{query}': {str(e)}"
            logging.error(error_msg)
            formatted_output += f"\n--- QUERY {query_idx + 1}: {query} ---\n"
            formatted_output += f"{error_msg}\n\n"
    
    if not all_results:
        logging.warning(f"No results found in Supabase for {len(queries)} queries with threshold {match_threshold}")
        return "No relevant results found in the knowledge base. Please try different search queries or adjust the similarity threshold."
    
    logging.info(f"Supabase search completed: {len(all_results)} total results from {len(queries)} queries")
    
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


##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.
    
    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, Tavily, or None)
        
    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # Configure Tavily search tool with metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
    elif search_api == SearchAPI.SUPABASE:
        # Configure Supabase search tool with metadata
        search_tool = supabase_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "supabase_search"  # Keep original name for trace clarity
        }
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []
        
    # Default fallback for unknown search API types
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.
    
    Args:
        response: The response object from Anthropic's API
        
    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # Return True if any web search requests were made
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False

def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.
    
    Args:
        response: The response object from OpenAI's API
        
    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    
    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # Step 2: Check provider-specific token limit patterns
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an OpenAI exception
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # Check for specific OpenAI error codes
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is a Google/Gemini exception
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # Check for specific Google API resource exhaustion patterns
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
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
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
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")

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
    # Previous approach removed only the matched portion, leaving one blank line per reference.
    # Consume the entire line (and its trailing newline) for any remaining ID lines to avoid
    # accumulating blank lines where references were listed, especially at document end.
    FULL_ID_LINE_PATTERN = r'(?m)^\s*\[?\^?\d+\]?\.?\sID:\s*sc_[^\s,)]+'  # matches full line with ID
    text = re.sub(FULL_ID_LINE_PATTERN + r'(?:\s*\n)?', '', text)
    text = re.sub(ID_PATTERN, '', text).strip()  # fallback (in-case of embedded patterns)

    # Collapse any runs of 3+ blank lines to just two (paragraph break) before footnotes.
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4) convert bare numeric refs like “[1]” → “[^^1]” -------------------
    text = re.sub(BARE_FOOTNOTE, r' [^\1] ', text).strip()

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

async def format_citations(text: str) -> str:
    """Formats citations in the text to use Markdown-style footnotes.
    
    Retrieves citation metadata from Supabase and converts sc_ids to proper
    footnote references with source information.
    
    Args:
        text: The input text containing sc_id citations
        
    Returns:
        The formatted text with deduplicated and renumbered footnotes
    """
    try:
        from supabase import create_client, Client
        
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logging.warning("SUPABASE_URL or SUPABASE_KEY not set, returning text without citation formatting")
            return text
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
    except ImportError:
        logging.warning("supabase-py package not installed, returning text without citation formatting")
        return text
    
    # Extract sc_ids and create initial footnotes
    out, footnote_map = md_cite_sc_ids(text)
    
    if not footnote_map:
        return text
    
    # Fetch reference data from Supabase
    try:
        # ref_data = supabase.table("ref_info").select(
        #     "sc_id, doc_title, page_start, page_end"
        # ).in_(
        #     "sc_id", list(footnote_map.keys())
        # ).execute().data
        ref_data = []
        sc_ids = list(footnote_map.keys())
        batch_size = 50

        for i in range(0, len(sc_ids), batch_size):
            batch = sc_ids[i:i + batch_size]
            batch_data = supabase.table("ref_info").select("sc_id, doc_title, page_start, page_end").in_("sc_id", batch).execute().data
            ref_data.extend(batch_data)
        
        # Replace sc_ids with formatted titles
        for each in ref_data:
            # Option to include page numbers (currently commented out):
            # formatted = f"{each['doc_title'].title()} p{each['page_start']}{'-'+str(each['page_end']) if each['page_start'] != each['page_end'] else ''}"
            # formatted = f"{each['doc_title'].title()}, {each['page_start']}"
            formatted = f"{each['doc_title'].title()}"
            out = out.replace(each['sc_id'], formatted)
        
    except Exception as e:
        logging.warning(f"Failed to fetch citation data from Supabase: {str(e)}")
    
    # Deduplicate and renumber footnotes
    out = dedupe_and_renumber_footnotes(out)
    return out