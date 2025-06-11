# Tests for code, completion, entity_completion, graph, ingestion, repo_processor, storage, summarization, temporal_awareness tasks
# Only basic import and callable checks, as most require complex setup

import pytest
import importlib

def import_task_module(module_path):
    return importlib.import_module(module_path)

def test_code_imports():
    import_task_module('cognee.tasks.code.enrich_dependency_graph_checker')
    import_task_module('cognee.tasks.code.get_local_dependencies_checker')
    import_task_module('cognee.tasks.code.get_repo_dependency_graph_checker')

def test_completion_imports():
    import_task_module('cognee.tasks.completion')

def test_entity_completion_imports():
    import_task_module('cognee.tasks.entity_completion')

def test_graph_imports():
    import_task_module('cognee.tasks.graph.extract_graph_from_code')
    import_task_module('cognee.tasks.graph.extract_graph_from_data')
    import_task_module('cognee.tasks.graph.extract_graph_from_data_v2')
    import_task_module('cognee.tasks.graph.infer_data_ontology')
    import_task_module('cognee.tasks.graph.models')

def test_ingestion_imports():
    import_task_module('cognee.tasks.ingestion.get_dlt_destination')
    import_task_module('cognee.tasks.ingestion.ingest_data')
    import_task_module('cognee.tasks.ingestion.migrate_relational_database')
    import_task_module('cognee.tasks.ingestion.resolve_data_directories')
    import_task_module('cognee.tasks.ingestion.save_data_item_to_storage')
    import_task_module('cognee.tasks.ingestion.transform_data')

def test_repo_processor_imports():
    import_task_module('cognee.tasks.repo_processor.get_local_dependencies')
    import_task_module('cognee.tasks.repo_processor.get_non_code_files')
    import_task_module('cognee.tasks.repo_processor.get_repo_file_dependencies')

def test_storage_imports():
    import_task_module('cognee.tasks.storage.add_data_points')
    import_task_module('cognee.tasks.storage.index_data_points')
    import_task_module('cognee.tasks.storage.index_graph_edges')

def test_summarization_imports():
    import_task_module('cognee.tasks.summarization.mock_summary')
    import_task_module('cognee.tasks.summarization.models')
    import_task_module('cognee.tasks.summarization.summarize_code')
    import_task_module('cognee.tasks.summarization.summarize_text')

def test_temporal_awareness_imports():
    import_task_module('cognee.tasks.temporal_awareness.build_graph_with_temporal_awareness')
    import_task_module('cognee.tasks.temporal_awareness.graphiti_model')
    import_task_module('cognee.tasks.temporal_awareness.index_graphiti_objects')
    import_task_module('cognee.tasks.temporal_awareness.search_graph_with_temporal_awareness')
