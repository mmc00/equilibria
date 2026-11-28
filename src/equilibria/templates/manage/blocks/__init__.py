"""MANAGE's model-specific blocks, by group of the design's equation inventory
(specs/templates/manage/changes/manage-template/design.md)."""

from equilibria.templates.manage.blocks.production import (
    PRODUCTION_FAMILIES,
    ManageProduction,
    production_group,
    production_tree,
)

__all__ = [
    "PRODUCTION_FAMILIES",
    "ManageProduction",
    "production_group",
    "production_tree",
]
