import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, FrozenSet, List, Set

try:
    import orjson as _json_backend
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _json_backend = None

import streamlit as st
from datasets import load_dataset


DATASET = "viktoroo/WildChat-1M-sampled-for-message-classification"

USER_CLASSIFICATION_COLUMN = "gpt-5-mini-use-case-response"
USER_CLASSIFICATION_SCHEMA = {
    "reference_to_specific_work": ["no", "yes"],
    "intent": ["transformative", "reproduce"]
}

ASSISTANT_CLASSIFICATION_COLUMN = "gpt-5-mini-refusal-response"
ASSISTANT_CLASSIFICATION_SCHEMA = {
    "assistant_refusal": ["no", "yes"],
    "refusal_type": [
        "capability_or_access",
        "copyright_or_source_restricted",
        "no_refusal",
        "other",
        "safety"
    ]
}


ExampleEntry = Dict[str, Any]
LabelValueIndex = Dict[str, FrozenSet[int]]
LabelInvertedIndex = Dict[str, LabelValueIndex]


@dataclass(frozen=True)
class ExampleIndex:
    entries: Dict[int, ExampleEntry]
    all_row_ids: FrozenSet[int]
    user_label_index: LabelInvertedIndex
    assistant_label_index: LabelInvertedIndex


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def _parse_label_payload(payload: str | None) -> Dict[str, str]:
    """Safely parse JSON payloads emitted by the classifiers."""
    payload = (payload or "").strip()
    if not payload:
        return {}
    try:
        if _json_backend is not None:
            return _json_backend.loads(payload)
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def _has_annotation(payload: str | None) -> bool:
    """Return True when a classification payload contains any data."""
    return bool((payload or "").strip())


def _row_has_annotation(row: Dict[str, Any]) -> bool:
    """Predicate used to filter dataset rows to only annotated examples."""
    return _has_annotation(row.get(USER_CLASSIFICATION_COLUMN)) and _has_annotation(
        row.get(ASSISTANT_CLASSIFICATION_COLUMN)
    )


def _freeze_label_index(
    label_index: DefaultDict[str, DefaultDict[str, Set[int]]]
) -> LabelInvertedIndex:
    """Convert mutable label indexes into hashable frozenset-backed structures."""
    return {
        label_name: {
            label_value: frozenset(row_ids)
            for label_value, row_ids in value_map.items()
        }
        for label_name, value_map in label_index.items()
    }


def _record_labels(
    row_idx: int,
    labels: Dict[str, str],
    target_index: DefaultDict[str, DefaultDict[str, Set[int]]],
) -> None:
    """Populate the inverted index for fast per-label lookups."""
    for label_name, label_value in labels.items():
        if not label_value:
            continue
        target_index[label_name][label_value].add(row_idx)


@st.cache_resource(show_spinner="Loading dataset and building example index...")
def load_index() -> ExampleIndex:
    logger.info("Loading dataset '%s' (split=train)...", DATASET)
    dataset = load_dataset(DATASET, split="train")
    required_columns = {
        "context",
        "user_message",
        "rest_of_conversation",
        "orig_idx",
        USER_CLASSIFICATION_COLUMN,
        ASSISTANT_CLASSIFICATION_COLUMN,
    }
    dataset = dataset.select_columns([col for col in dataset.column_names if col in required_columns])
    original_row_count = len(dataset)
    dataset = dataset.filter(_row_has_annotation)
    annotated_row_count = len(dataset)
    retained_pct = (annotated_row_count / original_row_count * 100) if original_row_count else 0
    logger.info(
        "Prefiltered annotated rows: kept %s/%s (%.2f%%).",
        annotated_row_count,
        original_row_count,
        retained_pct,
    )
    contexts = dataset["context"]
    user_messages = dataset["user_message"]
    rest_of_conversations = dataset["rest_of_conversation"]
    orig_indices = dataset["orig_idx"]
    user_label_payloads = dataset[USER_CLASSIFICATION_COLUMN]
    assistant_label_payloads = dataset[ASSISTANT_CLASSIFICATION_COLUMN]

    total_rows = len(user_messages)
    logger.info("Building cached index from %s rows...", total_rows)
    entries: Dict[int, ExampleEntry] = {}
    user_label_index: DefaultDict[str, DefaultDict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
    assistant_label_index: DefaultDict[str, DefaultDict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
    progress_interval = max(1, total_rows // 20)  # log roughly every 5%
    for row_idx in range(total_rows):
        context = contexts[row_idx] or []
        user_message = user_messages[row_idx]
        rest_of_conversation = rest_of_conversations[row_idx] or []
        immediate_assistant_response = rest_of_conversation[0] if rest_of_conversation else None
        entry = {
            "row_idx": row_idx,
            "orig_idx": orig_indices[row_idx],
            "context": context,
            "user_message": user_message,
            "immediate_assistant_response": immediate_assistant_response,
            "rest_of_conversation": rest_of_conversation,
            "user_labels": _parse_label_payload(user_label_payloads[row_idx]),
            "assistant_labels": _parse_label_payload(assistant_label_payloads[row_idx]),
        }
        entries[row_idx] = entry
        _record_labels(row_idx, entry["user_labels"], user_label_index)
        _record_labels(row_idx, entry["assistant_labels"], assistant_label_index)
        if (row_idx + 1) % progress_interval == 0 or row_idx + 1 == total_rows:
            logger.info("Indexed %s/%s rows", row_idx + 1, total_rows)
    logger.info("Finished building index with %s rows.", total_rows)
    return ExampleIndex(
        entries=entries,
        all_row_ids=frozenset(entries.keys()),
        user_label_index=_freeze_label_index(user_label_index),
        assistant_label_index=_freeze_label_index(assistant_label_index),
    )


def _build_filter_signature(user_filters: Dict[str, str], assistant_filters: Dict[str, str]) -> str:
    """Generate a stable signature for the active filters."""
    signature_payload = {
        "assistant": dict(sorted(assistant_filters.items())),
        "user": dict(sorted(user_filters.items())),
    }
    return json.dumps(signature_payload, sort_keys=True)


def _matching_rows(
    filters: Dict[str, str],
    inverted_index: LabelInvertedIndex,
    all_row_ids: FrozenSet[int],
) -> Set[int]:
    """Resolve the candidate row ids that satisfy the provided filters."""
    if not filters:
        return set(all_row_ids)
    matching_rows: Set[int] | None = None
    for label_name, label_value in filters.items():
        label_values = inverted_index.get(label_name)
        if not label_values:
            return set()
        row_ids = label_values.get(label_value)
        if not row_ids:
            return set()
        if matching_rows is None:
            matching_rows = set(row_ids)
        else:
            matching_rows.intersection_update(row_ids)
            if not matching_rows:
                return set()
    return matching_rows if matching_rows is not None else set(all_row_ids)


def filter_examples(index: ExampleIndex, user_filters: Dict[str, str], assistant_filters: Dict[str, str]) -> Set[int]:
    """Return row ids that satisfy both user and assistant label filters."""
    user_rows = _matching_rows(user_filters, index.user_label_index, index.all_row_ids)
    assistant_rows = _matching_rows(assistant_filters, index.assistant_label_index, index.all_row_ids)
    return user_rows & assistant_rows


def _render_schema_controls(schema: Dict[str, List[str]], key_prefix: str) -> Dict[str, str]:
    """Render select boxes for a schema and return selected values (excluding 'Any')."""
    selections: Dict[str, str] = {}
    for label_name, options in schema.items():
        label_title = label_name.replace("_", " ").capitalize()
        select_options = ["Any"] + options
        selection = st.selectbox(
            label_title,
            select_options,
            index=0,
            key=f"{key_prefix}-{label_name}",
        )
        if selection != "Any":
            selections[label_name] = selection
    return selections



def ui():
    st.title("WildChat Classification Viewer")
    st.caption(f"Visualizes results of annotation stored in "
               f"[{DATASET}](https://huggingface.co/datasets/viktoroo/WildChat-1M-sampled-for-message-classification)")

    index = load_index()

    st.sidebar.header("Annotation filters")
    with st.sidebar.expander("User classification", expanded=True):
        user_filters = _render_schema_controls(USER_CLASSIFICATION_SCHEMA, "user-schema")
    with st.sidebar.expander("Assistant classification", expanded=True):
        assistant_filters = _render_schema_controls(ASSISTANT_CLASSIFICATION_SCHEMA, "assistant-schema")

    matching_row_ids = filter_examples(index, user_filters, assistant_filters)
    filter_signature = _build_filter_signature(user_filters, assistant_filters)

    if "filter_signature" not in st.session_state:
        st.session_state["filter_signature"] = ""
    if "shuffled_row_ids" not in st.session_state:
        st.session_state["shuffled_row_ids"] = []
    if "current_example_pos" not in st.session_state:
        st.session_state["current_example_pos"] = 0

    if filter_signature != st.session_state["filter_signature"]:
        st.session_state["filter_signature"] = filter_signature
        shuffled_ids = list(matching_row_ids)
        random.shuffle(shuffled_ids)
        st.session_state["shuffled_row_ids"] = shuffled_ids
        st.session_state["current_example_pos"] = 0

    shuffled_ids = [
        row_id for row_id in st.session_state["shuffled_row_ids"] if row_id in matching_row_ids
    ]
    if len(shuffled_ids) != len(st.session_state["shuffled_row_ids"]):
        st.session_state["shuffled_row_ids"] = shuffled_ids

    ordered_examples = [index.entries[row_id] for row_id in st.session_state["shuffled_row_ids"]]

    total_examples = len(ordered_examples)

    if total_examples == 0:
        st.info("No examples found for the selected annotations yet.")
        return

    current_pos = st.session_state["current_example_pos"]
    if current_pos >= total_examples:
        current_pos = 0
        st.session_state["current_example_pos"] = 0

    navigation_row = st.container(border=True, horizontal=True)

    with navigation_row:
        if st.button("Previous", disabled=current_pos <= 0, width="stretch", type="primary"):
            st.session_state["current_example_pos"] = max(0, current_pos - 1)
            current_pos = st.session_state["current_example_pos"]
    with navigation_row:
        if st.button("Next", disabled=current_pos >= total_examples - 1, width="stretch", type="primary"):
            st.session_state["current_example_pos"] = min(total_examples - 1, current_pos + 1)
            current_pos = st.session_state["current_example_pos"]
    with navigation_row:
        if st.button("Reshuffle order", disabled=total_examples < 2):
            random.shuffle(st.session_state["shuffled_row_ids"])
            st.session_state["current_example_pos"] = 0
            ordered_examples = [index.entries[row_id] for row_id in st.session_state["shuffled_row_ids"]]
    with navigation_row:
        st.markdown(f"Showing example {current_pos + 1} out of {total_examples}.\n    "
                    f"Classified message is marked with ⭐")

    current_example = ordered_examples[current_pos]
    st.subheader("Selected example")

    for turn in current_example["context"]:
        st.caption(f"{turn['role']}:")
        st.container(border=True).markdown(turn['content'])

    st.caption("⭐ Classified user message:")
    st.container(border=True).markdown(current_example["user_message"])

    for turn in current_example["rest_of_conversation"]:
        st.caption(f"{turn['role']}:")
        st.container(border=True).markdown(turn['content'])


if __name__ == "__main__":
    ui()
