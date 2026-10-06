"""Entity-text loading, cleaning, downloading, and alignment for FB15k-237.

Text descriptions must be aligned to the exact entity IDs created by
preprocessing.dataset.KGDataset.entity2id.

This module never creates a second entity vocabulary. It only maps external
text keyed by the original FB15k-237 entity strings onto the existing KG
integer IDs.

Text priority:
    1. Long entity description
    2. Short entity text/name
    3. Missing text marker

The helper downloader makes Phase 2 reproducible in fresh environments such
as Google Colab without requiring the raw text files to be committed to Git.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# External FB15k-237 entity-text resources
# ---------------------------------------------------------------------------

SHORT_TEXT_URL = (
    "https://raw.githubusercontent.com/yao8839836/"
    "kg-bert/master/data/FB15k-237/entity2text.txt"
)

LONG_TEXT_URL = (
    "https://huggingface.co/datasets/KGraph/FB15k-237/"
    "resolve/main/data/FB15k_mid2description.txt"
)


@dataclass
class EntityTextAlignment:
    """Entity descriptions aligned to KG entity IDs.

    Attributes
    ----------
    texts_by_id:
        Entity text indexed by the SAME integer entity ID used by the KG.

        Example:

            texts_by_id[57]

        corresponds to the entity whose KG representation is:

            entity_repr[57]

    has_text:
        Boolean flag indicating whether usable text was found for each
        entity ID.

    matched:
        Number of KG entities for which usable text was found.

    missing:
        Number of KG entities for which no usable text was found.
    """

    texts_by_id: list[str]
    has_text: list[bool]
    matched: int
    missing: int

    @property
    def coverage(self) -> float:
        """Return entity-text coverage as a fraction between 0 and 1."""

        total = len(self.texts_by_id)

        return (
            self.matched / total
            if total
            else 0.0
        )


def download_entity_text_files(
    text_dir: str | Path,
) -> tuple[Path, Path]:
    """Download FB15k-237 entity-text resources if they are missing.

    Downloads
    ---------
    Short text:
        entity2text.txt

    Long descriptions:
        FB15k_mid2description.txt

    Existing files are not downloaded again.

    Parameters
    ----------
    text_dir:
        Directory in which the text resources should be stored.

    Returns
    -------
    tuple[Path, Path]
        (long_text_path, short_text_path)
    """

    text_dir = Path(text_dir)
    text_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    short_path = (
        text_dir
        / "entity2text.txt"
    )

    long_path = (
        text_dir
        / "FB15k_mid2description.txt"
    )

    if not short_path.exists():
        print(
            "[entity-text] "
            "downloading short entity text..."
        )

        urllib.request.urlretrieve(
            SHORT_TEXT_URL,
            short_path,
        )

    if not long_path.exists():
        print(
            "[entity-text] "
            "downloading long entity descriptions..."
        )

        urllib.request.urlretrieve(
            LONG_TEXT_URL,
            long_path,
        )

    return long_path, short_path


def clean_entity_text(
    text: str,
) -> str:
    """Clean raw FB15k entity text before sending it to the LM.

    Long-description files may contain values such as:

        "Some description.\\nMore text."@en

    This function converts them into clean text such as:

        Some description. More text.
    """

    text = text.strip()

    # Remove language suffix.
    if text.endswith("@en"):
        text = text[:-3].strip()

    # Decode JSON-style quoted strings safely.
    #
    # Example:
    #
    #   "Some text.\\nMore text."
    #
    # becomes:
    #
    #   Some text.
    #   More text.
    #
    if (
        len(text) >= 2
        and text.startswith('"')
        and text.endswith('"')
    ):
        try:
            text = json.loads(text)

        except json.JSONDecodeError:
            # Safe fallback for unusual malformed rows.
            text = text[1:-1]

    # Convert remaining literal or real line breaks to spaces.
    text = text.replace(
        "\\n",
        " ",
    )

    text = text.replace(
        "\n",
        " ",
    )

    text = text.replace(
        "\r",
        " ",
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def read_text_mapping(
    path: str | Path,
) -> dict[str, str]:
    """Read a tab-separated `entity_key<TAB>text` file.

    The original FB15k-237 Freebase entity key is preserved exactly and used
    only for lookup against the existing KG `entity2id` vocabulary.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Entity-text file not found: {path}"
        )

    mapping: dict[str, str] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.rstrip("\n")

            if not line.strip():
                continue

            # Only split on the first tab because the description itself may
            # contain additional text content.
            parts = line.split(
                "\t",
                1,
            )

            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no}: expected "
                    "`entity<TAB>text`"
                )

            entity_key = (
                parts[0].strip()
            )

            text = clean_entity_text(
                parts[1]
            )

            if not entity_key:
                raise ValueError(
                    f"{path}:{line_no}: "
                    "empty entity key"
                )

            # Keep the first occurrence so the result remains deterministic.
            if entity_key not in mapping:
                mapping[entity_key] = text

    return mapping


def align_entity_texts(
    entity2id: dict[str, int],
    long_text_path: str | Path,
    short_text_path: str | Path | None = None,
) -> EntityTextAlignment:
    """Align external entity text to the existing KG entity IDs.

    Priority
    --------
    1. Long description
    2. Short entity text/name
    3. Empty string + has_text=False

    Important
    ---------
    No new entity IDs are created.

    If:

        entity2id["/m/example"] == 57

    then:

        texts_by_id[57]

    will contain the text for that SAME entity.
    """

    long_text = read_text_mapping(
        long_text_path
    )

    short_text: dict[str, str] = {}

    if short_text_path is not None:
        short_text = read_text_mapping(
            short_text_path
        )

    num_entities = len(
        entity2id
    )

    texts_by_id = [
        ""
    ] * num_entities

    has_text = [
        False
    ] * num_entities

    matched = 0

    for (
        entity_key,
        entity_id,
    ) in entity2id.items():

        # First choice: long semantic description.
        text = long_text.get(
            entity_key,
            "",
        ).strip()

        # Fallback: short readable entity text/name.
        if not text:
            text = short_text.get(
                entity_key,
                "",
            ).strip()

        if text:
            texts_by_id[
                entity_id
            ] = text

            has_text[
                entity_id
            ] = True

            matched += 1

    missing = (
        num_entities
        - matched
    )

    return EntityTextAlignment(
        texts_by_id=texts_by_id,
        has_text=has_text,
        matched=matched,
        missing=missing,
    )