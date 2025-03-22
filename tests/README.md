# LightRAG Tests

This directory contains tests for the LightRAG project, including unit tests and integration tests for various features.

## Test Files

- `test_lightrag_ollama_chat.py`: Tests LightRAG's Ollama compatibility interface
- `test_path_retrieval.py`: Unit tests for PathRAG integration and path-based retrieval algorithms
- `test_path_integration.py`: Integration tests for PathRAG functionality with a simple knowledge graph

## Running Tests

### Prerequisites

Make sure you have pytest and its dependencies installed:

```bash
pip install pytest pytest-asyncio
```

### Running Unit Tests for Path-based Retrieval

```bash
# Run all tests in the path retrieval test file
pytest -xvs test_path_retrieval.py

# Run specific test class
pytest -xvs test_path_retrieval.py::TestPathFinding

# Run a specific test
pytest -xvs test_path_retrieval.py::TestPathFinding::test_find_paths_and_edges
```

### Running Path Integration Tests

```bash
# Run the path integration tests
pytest -xvs test_path_integration.py
```

### Running All Tests

```bash
# Run all tests
pytest -xvs
```

## Test Details

### Path Retrieval Unit Tests

- Tests for path finding algorithms
- Tests for weighted path exploration
- Tests for path-based querying
- Tests for integration with the main LightRAG class

### Path Integration Tests  

This integration test:
1. Creates a mock LightRAG instance
2. Inserts sample data with relationships
3. Runs queries using different modes (global, path, hybrid)
4. Compares the results to verify path-based retrieval functionality