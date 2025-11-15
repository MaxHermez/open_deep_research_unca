from pydantic import Field, BaseModel
from typing_extensions import Annotated
from langchain_core.tools import tool, InjectedToolArg, BaseToolkit


def get_search_results(query: str, text: str) -> list[dict]:
    context_size = 200
    results = []
    searching = True
    curr_text = text
    removed = 0
    while searching:
        if query in curr_text:
            start = max(curr_text.index(query) - context_size, 0)
            end = min(curr_text.index(query) + len(query) + context_size, len(curr_text))
            results.append({"text": curr_text[start:end], "t_index": curr_text.index(query) + removed})
            curr_text = curr_text[end:]
            removed += end
        else:
            searching = False

    return results

class FindTextArgs(BaseModel):
    query: str = Field(..., description="The substring to find")
    text: Annotated[str, InjectedToolArg] = Field(..., description="The text to search within")

@tool(
    description="Find a specific substring within the text",
    args_schema=FindTextArgs,
)
def find_text(
    query: str,
    text: Annotated[str, InjectedToolArg]
) -> dict:
    """Find a specific substring within the text.

    Args:
        query (str): The substring to find.
        text (Annotated[str, InjectedToolArg]): The text to search within.

    Returns:
        dict: A dictionary containing the search results.
    """
    results = get_search_results(query, text)
    return {"results": results}

class ReplaceTextArgs(BaseModel):
    query: str = Field(..., description="The substring to replace")
    replacement: str = Field(..., description="The new substring to replace the old one")
    text: Annotated[str, InjectedToolArg] = Field(..., description="The text to modify")

@tool(
    description="Replace a specific substring within the text with a new substring",
    args_schema=ReplaceTextArgs,
)
def replace_text(
    query: str,
    replacement: str,
    text: Annotated[str, InjectedToolArg]
) -> dict:
    """Replace a specific substring within the text with a new substring.

    Args:
        query (str): The substring to replace.
        replacement (str): The new substring to replace the old one.
        text (Annotated[str, InjectedToolArg]): The text to modify.

    Returns:
        dict: A dictionary containing the modified text.
    """
    return {
        "text": text.replace(query, replacement),
    }

class GetFullTextArgs(BaseModel):
    text: Annotated[str, InjectedToolArg] = Field(..., description="The text to retrieve.")

@tool(
    description="Get a specific line from the text",
    args_schema=GetFullTextArgs,
)
def get_full_text(
    text: Annotated[str, InjectedToolArg]
) -> dict:
    """Get the full text.

    Args:
        text (Annotated[str, InjectedToolArg]): The text to retrieve.

    Returns:
        dict: A dictionary containing the full text.
    """
    return {
        "text": text
    }

class GetTextLinesArgs(BaseModel):
    start_line: int = Field(..., description="The starting line number (0-indexed).")
    end_line: int = Field(..., description="The ending line number (0-indexed).")
    text: Annotated[str, InjectedToolArg] = Field(..., description="The text to search within.")

@tool(
    description="Get a specific range of lines from the text",
    args_schema=GetTextLinesArgs,
)
def get_text_lines(
    start_line: int,
    end_line: int,
    text: Annotated[str, InjectedToolArg]
) -> dict:
    """Get a specific range of lines from the text.

    Args:
        start_line (int): The starting line number (0-indexed).
        end_line (int): The ending line number (0-indexed).
        text (Annotated[str, InjectedToolArg]): The text to search within.

    Returns:
        dict: A dictionary containing the requested lines.
    """
    return {
        "text": text.splitlines()[start_line:end_line]
    }

# define the toolkit
text_tools = [find_text, replace_text, get_full_text, get_text_lines]