## ADDED Requirements

### Requirement: One data folder
Source: ADR-0010

Every file that is not equilibria code and that a model reads or is compared
against must live under `data/`, in the folder of the dataset (or model) it
belongs to; `src/` and `tests/` must not hold such files.

### Requirement: Every reference says where it came from
Source: ADR-0010, issue #23

Every folder of reference results under `data/` must carry a `source.yaml`
recording the dataset, reference model, variant, tool and version, command,
date and the sha256 of its inputs; a test must fail when one is missing, when
a verified result's inputs changed after it was generated, or when a result
is not verified.

### Requirement: A reference is requested by what it is
Source: ADR-0010

Tests must obtain reference results by dataset, tool and variant, and the
request must fail when no recorded variant matches it.

## MODIFIED Requirements

## REMOVED Requirements
