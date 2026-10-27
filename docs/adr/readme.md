# Architecture Decision Records

Decisions that are expensive to reverse, surprising without context, and the
result of a real trade-off. ADRs are immutable: when a decision changes, a new
ADR supersedes the old one. Create one with
`python scripts/dev/spec_tools.py new-adr <slug>`.

| ADR | Decision |
| --- | --- |
| [0001](0001-blocks-are-the-model.md) | Blocks are the model; a monolith is only a starting point and an oracle |
| [0002](0002-driver-prepares-and-solves.md) | The driver prepares and solves; it holds no model logic |
| [0003](0003-shock-lives-in-shock-block.md) | The shock lives in the ShockBlock and is written as in GEMPACK |
| [0004](0004-overwrite-one-block-per-exercise.md) | An exercise changes one block with `@overwrite` and leaves the others alone |
| [0005](0005-one-name-one-declaration.md) | One name, one declaration; each block declares what it uses |
| [0006](0006-blocks-do-not-know-periods.md) | Blocks do not know about periods; the SP copy adds the time index |
| [0007](0007-no-block-registry.md) | No block registry |
| [0008](0008-reference-models-are-the-source-of-truth.md) | GAMS and GEMPACK are the source of truth; fidelity outranks match % |
| [0009](0009-one-name-dictionary-per-model.md) | One name dictionary per model, checked for completeness against both sides |

Vocabulary: [`GLOSSARY.md`](../../GLOSSARY.md).
