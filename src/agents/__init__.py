"""Agent module initialization."""

from .agent_factory import AgentFactory
from .file_type_agent import FileTypeAgent
from .format_detector_agent import FormatDetectorAgent
from .data_extraction_agent import DataExtractionAgent

# Register all agents
AgentFactory.register("FileTypeAgent", FileTypeAgent)
AgentFactory.register("FormatDetectorAgent", FormatDetectorAgent)
AgentFactory.register("DataExtractionAgent", DataExtractionAgent)
