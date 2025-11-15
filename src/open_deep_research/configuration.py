from pydantic import BaseModel, Field
from typing import Any, List, Optional
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
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

# class GoogleDocsConfig(BaseModel):
#     output_to_google_docs: bool = Field(
#         default=False,
#         metadata={
#             "x_oap_ui_config": {
#                 "type": "boolean",
#                 "default": False,
#                 "description": "Whether to output the final report to Google Docs. If enabled, the final report will be saved to Google Drive."
#             }
#         }
#     )
#     drive_output_directory: str = Field(
#         default="odr_output",
#         metadata={
#             "x_oap_ui_config": {
#                 "type": "text",
#                 "default": "odr_output",
#                 "description": "The name of the output directory in Google Drive where research results will be stored."
#             }
#         }
#     )
#     output_file_name: str = Field(
#         default="research_results",
#         metadata={
#             "x_oap_ui_config": {
#                 "type": "text",
#                 "default": "research_results",
#                 "description": "The base name of the output file in Google Drive. The file will be saved as <output_file_name>_<timestamp>.txt."
#             }
#         }
#     )

class Configuration(BaseModel):
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
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
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
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
        metadata={
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
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
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
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=15000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 15000,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=32768,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 32768,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=15000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 15000,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=32768,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 32768,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    best_draft_model: str = Field(
        default="openai:o3-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:o3-mini",
                "description": "Model for selecting the best draft from multiple narrative drafts"
            }
        }
    )
    best_draft_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for best draft model"
            }
        }
    )
    format_citations: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to format citations in the final report . If enabled, the final report will include citations in the format [sc_id]."
            }
        }
    )

    draft_template: Optional[str] = Field(
        default="none",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "none",
                "description": "Select a draft template to prepend to the research brief. This will include the template content as context before conducting research.",
                "options": get_draft_options()
            }
        }
    )

    qualitative_analysis: Optional[str] = Field(
        default="none",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "none",
                "description": "Select a qualitative analysis method to apply to the research findings.",
                "options": get_qual_analysis_options()
            }
        }
    )

    narrative_drafts: int = Field(
        default=3,
        metadata={
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
    # google_docs_output_config: Optional[GoogleDocsConfig] = Field(
    #     default=None,
    #     optional=True,
    #     metadata={
    #         "x_oap_ui_config": {
    #             "type": "google_docs",
    #             "description": "Google Docs output configuration. If provided, the final report will be saved to Google Drive."
    #         }
    #     }
    # )

    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
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

    def get_draft_template_content(self) -> Optional[str]:
        """Get the content of the selected draft template (case/underscore-insensitive)."""
        if not self.draft_template or str(self.draft_template).lower() == "none":
            return None

        def norm(s: str) -> str:
            # Normalize by treating underscores and spaces equivalently and ignoring case
            return " ".join(str(s).replace("_", " ").lower().split())

        try:
            current_dir = Path(__file__).parent
            drafts_dir = current_dir / "drafts"
            target = norm(self.draft_template)

            for file_path in drafts_dir.glob("*.md"):
                if norm(file_path.stem) == target:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read()
            return None
        except Exception:
            return None

    def get_qualitative_analysis_content(self) -> Optional[str]:
        """Get the content of the selected qualitative analysis method (case/underscore-insensitive)."""
        if not self.qualitative_analysis or str(self.qualitative_analysis).lower() == "none":
            return None

        def norm(s: str) -> str:
            # Normalize by treating underscores and spaces equivalently and ignoring case
            return " ".join(str(s).replace("_", " ").lower().split())

        try:
            current_dir = Path(__file__).parent
            fault_trees_dir = current_dir / "fault_trees"
            target = norm(self.qualitative_analysis)

            for file_path in fault_trees_dir.glob("*.md"):
                if norm(file_path.stem) == target:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read()
            return None
        except Exception:
            return None

    class Config:
        arbitrary_types_allowed = True