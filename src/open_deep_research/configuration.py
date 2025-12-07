from pydantic import BaseModel, Field
from typing import Any, List, Optional, cast, Dict
from langchain_core.runnables import RunnableConfig
import os
from enum import Enum
from pathlib import Path

def _get_dir_options(dirname: str):
    """Get available draft options from the drafts folder."""
    try:
        # Get the current file's directory and navigate to the drafts folder
        current_dir = Path(__file__).parent
        drafts_dir = current_dir / dirname
        
        if not drafts_dir.exists():
            return [{"label": "None", "value": "none"}]
        
        # Get all .md files in the drafts directory
        draft_files = list(drafts_dir.glob("*.md"))
        
        if not draft_files:
            return [{"label": "None", "value": "none"}]
        
        # Create options list with file names without .md extension
        options = [{"label": "None", "value": "none"}]  # Default option
        for file_path in draft_files:
            file_name = file_path.stem  # Gets filename without extension
            options.append({"label": file_name, "value": file_name.lower().replace(" ", "_")})
        print(f"Draft options: {options}")  # Debugging output
        return options
    except Exception:
        return [{"label": "None", "value": "none"}]

def get_draft_options():
    """Get available draft options."""
    return _get_dir_options("drafts")

def get_qual_analysis_options():
    """Get available qualitative analysis options."""
    return _get_dir_options("fault_trees")

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    SUPABASE = "supabase"
    NONE = "none"

class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=10,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.SUPABASE,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "select",
                "default": "supabase",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "Supabase RAG Search", "value": SearchAPI.SUPABASE.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=15000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 15000,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=32768,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 32768,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=15000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 15000,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=32768,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 32768,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    best_draft_model: str = Field(
        default="openai:o3-mini",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:o3-mini",
                "description": "Model for selecting the best draft from multiple narrative drafts"
            }
        }
    )
    best_draft_model_max_tokens: int = Field(
        default=10000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for best draft model"
            }
        }
    )
    format_citations: bool = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to format citations in the final report . If enabled, the final report will include citations in the format [sc_id]."
            }
        }
    )

    draft_template: Optional[str] = Field(
        default="none",
        json_schema_extra=cast(Dict[str, Any], {
            "x_oap_ui_config": {
                "type": "select",
                "default": "none",
                "description": "Select a draft template to prepend to the research brief. This will include the template content as context before conducting research.",
                "options": get_draft_options()
            }
        })
    )

    qualitative_analysis: Optional[str] = Field(
        default="none",
        json_schema_extra=cast(Dict[str, Any], {
            "x_oap_ui_config": {
                "type": "select",
                "default": "none",
                "description": "Select a qualitative analysis method to apply to the research findings.",
                "options": get_qual_analysis_options()
            }
        })
    )

    narrative_drafts: int = Field(
        default=3,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 0,
                "max": 10,
                "step": 1,
                "description": "Number of narrative drafts to generate for the final report. The final report will be a combination of these drafts."
            }
        }
    )

    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        arbitrary_types_allowed = True