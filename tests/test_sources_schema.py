"""Level-1 (pure logic, no LLM/MCP) tests for the sources.json contract.

The schema being enforced is the one in sourcing-subagent-deep-dive.md — the
handoff contract between sourcing-agent and the orchestrator/dataset-agent.
"""

import json
from pathlib import Path

import pytest

from tools.sources_schema import (
    VALID_STATUSES,
    coverage_summary,
    find_duplicate_dataset_ids,
    validate_sources,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _budget() -> dict:
    return json.loads((FIXTURES / "class_budget.json").read_text())


def _valid_source(**overrides) -> dict:
    base = {
        "source": "roboflow_universe",
        "dataset_id": "traffic-detection/traffic-detection-4",
        "url": "https://universe.roboflow.com/traffic-detection/traffic-detection-4",
        "classes_covered": ["car", "truck"],
        "image_count": 1200,
        "annotation_format": "YOLO",
        "license": "CC BY 4.0",
        "annotation_coverage": {"car": 800, "truck": 400},
        "quality_notes": "mixed lighting, some occlusion",
        "status": "available",
    }
    return {**base, **overrides}


class TestValidateSources:
    def test_valid_document_passes(self):
        assert validate_sources([_valid_source()]) == []

    def test_non_list_document_rejected(self):
        errors = validate_sources({"source": "kaggle"})
        assert len(errors) == 1
        assert "JSON array" in errors[0]

    def test_empty_list_is_valid(self):
        assert validate_sources([]) == []

    def test_missing_required_field_flagged(self):
        source = _valid_source()
        del source["license"]
        errors = validate_sources([source])
        assert any("license" in e for e in errors)

    def test_quality_notes_is_optional(self):
        source = _valid_source()
        del source["quality_notes"]
        assert validate_sources([source]) == []

    def test_invalid_status_flagged(self):
        errors = validate_sources([_valid_source(status="downloaded")])
        assert any("status" in e for e in errors)

    def test_all_valid_statuses_accepted(self):
        for status in VALID_STATUSES:
            assert validate_sources([_valid_source(status=status)]) == []

    def test_coverage_key_outside_classes_covered_flagged(self):
        source = _valid_source(annotation_coverage={"car": 800, "bus": 10})
        errors = validate_sources([source])
        assert any("bus" in e for e in errors)

    def test_negative_image_count_flagged(self):
        errors = validate_sources([_valid_source(image_count=-5)])
        assert any("image_count" in e for e in errors)

    def test_error_messages_include_entry_index(self):
        bad = _valid_source(status="nope")
        errors = validate_sources([_valid_source(), bad])
        assert any("[1]" in e for e in errors)


class TestFindDuplicateDatasetIds:
    def test_no_duplicates(self):
        sources = [_valid_source(), _valid_source(dataset_id="other/ds")]
        assert find_duplicate_dataset_ids(sources) == []

    def test_duplicates_reported_once(self):
        sources = [_valid_source(), _valid_source(), _valid_source()]
        assert find_duplicate_dataset_ids(sources) == [
            "traffic-detection/traffic-detection-4"
        ]


class TestCoverageSummary:
    def test_annotated_counts_summed_across_available_sources(self):
        sources = [
            _valid_source(),
            _valid_source(
                dataset_id="another/cars",
                classes_covered=["car"],
                annotation_coverage={"car": 300},
            ),
        ]
        summary = coverage_summary(sources, _budget())
        assert summary["car"]["annotated"] == 1100
        assert summary["car"]["target"] == 500
        assert summary["car"]["met"] is True

    def test_needs_annotation_counts_as_raw_not_annotated(self):
        sources = [
            _valid_source(
                dataset_id="kaggle/raw-buses",
                source="kaggle",
                classes_covered=["bus"],
                image_count=300,
                annotation_format="none",
                annotation_coverage={"bus": 0},
                status="needs_annotation",
            )
        ]
        summary = coverage_summary(sources, _budget())
        assert summary["bus"]["annotated"] == 0
        assert summary["bus"]["raw"] == 300
        assert summary["bus"]["met"] is False

    def test_uncovered_class_has_zero_counts_and_full_gap(self):
        summary = coverage_summary([], _budget())
        assert summary["truck"] == {
            "target": 300,
            "annotated": 0,
            "raw": 0,
            "gap": 300,
            "met": False,
        }

    def test_gap_never_negative(self):
        summary = coverage_summary([_valid_source()], _budget())
        assert summary["car"]["gap"] == 0

    def test_inputs_not_mutated(self):
        sources = [_valid_source()]
        budget = _budget()
        snapshot = json.dumps([sources, budget], sort_keys=True)
        coverage_summary(sources, budget)
        validate_sources(sources)
        assert json.dumps([sources, budget], sort_keys=True) == snapshot
