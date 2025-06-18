AGENTS.md

This document describes the complete file structure of the test application and explains the role of each component in the format detection pipeline for SE3.0. The goal of this repository is to provide a minimal, self-contained environment for developing and validating the core logic of document format detection, field mapping, and user feedback handling without the overhead of the full production backend.

Project Layout

test_app/
├── src/
│   ├── agents/
│   │   ├── dependency_agent.py
│   │   ├── file_type_agent.py
│   │   └── format_detector_agent.py
│   │
│   ├── api/
│   │   ├── feedback.py
│   │   ├── main.py
│   │   └── upload.py
│   │
│   ├── data/
│   │   └── sample_files/
│   │
│   ├── models/
│   │   ├── manager.py
│   │   └── master_formats.xlsx
│   │
│   ├── orchestrator/
│   │   ├── core_orchestrator.py
│   │   └── document_pipeline.py
│   │
│   └── utils/
│       ├── embeddings.py
│       └── preprocessing.py
│
└── tests/
    ├── test_agents.py
    └── test_format_detection.py

src/agents

The agents directory contains the core components that perform individual steps of the document processing pipeline.
	•	file_type_agent.py checks that the uploaded file is one of the supported types (CSV or Excel) by attempting a lightweight read and validating the extension. It rejects unsupported or corrupted files early.
	•	format_detector_agent.py implements the semantic matching logic: it reads the headers of an input file, computes embeddings via a Sentence Transformer, and compares them against examples in the master_formats.xlsx. It returns the most likely format name and a confidence score.
	•	dependency_agent.py is a placeholder for any future logic that might infer parent/child relationships (such as employee–dependent data) from the document structure, but is not required for the current test flows.

src/orchestrator

The orchestrator folder drives the end-to-end workflow:
	•	document_pipeline.py defines a simple pipeline that takes an uploaded file, runs the FileTypeAgent, loads it into a DataFrame, and then invokes the FormatDetectorAgent. This script is ideal for quick integration tests.
	•	core_orchestrator.py builds on the base pipeline by adding a confidence threshold check and a stubbed fallback to an external LLM (via LangChain) when semantic matching confidence is low. It also handles basic caching logic for format lookups.

src/api

The api directory houses the FastAPI interface that clients use to interact with the test application:
	•	main.py initializes the FastAPI app and includes middleware for logging and error handling.
	•	upload.py defines the /upload endpoint, which accepts a file and triggers the orchestrator pipeline.
	•	feedback.py implements /feedback and /correction endpoints allowing users to submit their feedback or corrections. Feedback is stored locally in JSON for now and will be wired to a database once the schema is confirmed.

src/models

The models directory holds reference data for format detection and the logic to manage pre-trained models:
	•	master_formats.xlsx contains the list of known document formats and their sample headers or descriptors. This Excel file is the single source of truth for format matching.
	•	manager.py is a small utility that ensures the Sentence Transformer model is available on disk—downloading it on demand if missing—and provides caching of embeddings for faster repeated runs.

src/utils

The utils module contains helper functions:
	•	preprocessing.py includes a minimal cleanup routine to trim whitespace, drop empty rows, and standardize basic formatting before semantic matching.
	•	embeddings.py wraps the Sentence Transformer loading and embedding generation logic to keep agent code concise.

src/data

The data folder is reserved for sample input files used during development and testing. Place any Excel or CSV examples under sample_files to drive local experiments.

tests

The tests directory contains unit tests to validate core functionality in isolation:
	•	test_format_detection.py verifies that the FormatDetectorAgent returns expected format matches for known samples.
	•	test_agents.py includes smoke tests for each agent to ensure they can be instantiated and run on placeholder data.

⸻

This structure keeps only the essential pieces needed for the test application—format detection, orchestration, API stubs, and sample data—while removing any extra components or dependencies from the full SE3.0 backend. Feel free to adjust paths or filenames as you refine the test flows.
