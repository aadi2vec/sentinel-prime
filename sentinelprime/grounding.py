"""Document coverage: which of the workspace's documents did the run ever open?

In due diligence the dominant failure is not faulty reasoning, it is never opening a
document — a failure with no reasoning in it at all, and therefore one that costs neither
a judge nor a token to detect. Everything else in this repo needs an LM to have an opinion;
this needs a substring.

Name-mention is the proxy: a filename appearing anywhere in the run's code or interpreter
output counts, so a directory listing marks every file it prints as touched. That is
deliberately permissive. Coverage is a diagnostic, and over-reporting missed documents
would bury the real ones in noise, so it bounds the failure from one side only — a
document whose name never appears was certainly not read.

The companion check — did the run read the *facts* a criterion names, or did the model
merely assert them — is `credit.grounding_from_trajectory`, kept next to the credit path
it feeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CoverageReport:
    documents: list[str]
    touched: list[str]
    untouched: list[str]

    @property
    def rate(self) -> float:
        # No documents is complete coverage, not a division error: a task with nothing to
        # read cannot have missed anything.
        return len(self.touched) / len(self.documents) if self.documents else 1.0


def document_names(documents_dir: str | Path) -> list[str]:
    """Filenames directly under `documents_dir`; empty when it does not exist."""
    path = Path(documents_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_file())


def document_coverage(trajectory: list[dict],
                      documents: Iterable[str]) -> CoverageReport:
    """Which documents the trajectory ever named, in the order given.

    A name mentioned anywhere in the run's code or interpreter output counts, so a
    directory listing marks every file it prints as touched. That is deliberately
    permissive: coverage is a diagnostic, and over-reporting missed documents would bury
    the real ones in noise. It bounds the failure from one side — a document whose name
    never appears was certainly not read.
    """
    haystack = "\n".join(
        f"{step.get('code') or ''}\n{step.get('output') or ''}" for step in trajectory
    )
    docs = list(documents)
    touched = [name for name in docs if name and name in haystack]
    return CoverageReport(
        documents=docs,
        touched=touched,
        untouched=[name for name in docs if name not in touched],
    )
