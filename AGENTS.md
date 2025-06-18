# Agent Overview

This repository contains a small collection of agents that cooperate to detect and process structured document formats. The agents live in `backend/agents` and are orchestrated by the pipelines in `backend/app/orchestrator`.

## 1. FileTypeAgent
- **Purpose**: Performs a quick validation of the uploaded file type.
- **Functionality**: Attempts lightweight reads of the file to verify that the extension matches the actual content (CSV, Excel, PDF or image). Unsupported or unreadable files are rejected early in the pipeline.
- **Reference**: [`backend/agents/core/file_type.py`](backend/agents/core/file_type.py)

## 2. FormatDetectorAgent
- **Purpose**: Determines which document format definition best matches the uploaded file.
- **Functionality**: Cleans the dataframe and uses a mix of semantic similarity, pattern matching and optional LLM analysis to score available formats. It returns the most likely format ID and confidence score.
- **Reference**: [`backend/agents/core/detector.py`](backend/agents/core/detector.py)

## 3. Core Agent (LLM fallback)
- **Purpose**: When local detection confidence is low, this agent can query external LLMs through LangChain to infer the format.
- **Functionality**: Takes the raw document sample and calls the configured LLM (e.g. Claude) to classify the format, returning the best match from the database.
- **Note**: The high level orchestration that manages caching and this fallback logic lives in [`backend/app/orchestrator/core_orchestrator.py`](backend/app/orchestrator/core_orchestrator.py).

## 4. Format‑specific Agents
- **Purpose**: Once a format is known, a dedicated agent extracts structured fields.
- **Examples**:
  - `CensusAgent` – processes census style spreadsheets and extracts fields such as first name, last name and date of birth.
  - Other formats such as insurance statements can be implemented in a similar fashion under `backend/agents/format_specific/`.
- **Reference**: [`backend/agents/format_specific/census_agent.py`](backend/agents/format_specific/census_agent.py)

## 5. FeedbackAgent
- **Purpose**: Captures user corrections on extracted data for future learning.
- **Functionality**: Stores submitted feedback via the cache layer so models can be retrained or fine‑tuned later.
- **Reference**: [`backend/agents/feedback.py`](backend/agents/feedback.py)

## 6. AgentFactory
- **Purpose**: Central registry for creating agent instances by name.
- **Functionality**: Maps friendly names ("CensusAgent", "FormatDetectorAgent", etc.) to their classes and instantiates them on demand. New agents should be added here so the orchestrators can load them.
- **Reference**: [`backend/agents/factory.py`](backend/agents/factory.py)

## Orchestration
The simplest pipeline is implemented in `DocumentProcessingOrchestrator` (`backend/app/orchestrator/document_pipeline.py`). It validates file type, loads the dataframe and asks `FormatDetectorAgent` to identify the format. The higher level `CoreAgentOrchestrator` adds caching and invokes the appropriate format‑specific agent for extraction.
