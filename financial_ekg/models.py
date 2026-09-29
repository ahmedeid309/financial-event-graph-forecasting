from __future__ import annotations

from dataclasses import dataclass

@dataclass
class GraphNode:
    """Serializable graph node row.

    Attributes:
        node_id: Stable node identifier used by edges.
        node_type: Semantic node type, such as article, event, company, or date.
        name: Human-readable node label.
        ticker: Optional stock ticker associated with the node.
        date: Optional ISO date associated with the node.
        attributes_json: JSON string with extra node attributes.
    """

    node_id: str
    node_type: str
    name: str
    ticker: str = ""
    date: str = ""
    attributes_json: str = "{}"


@dataclass
class GraphEdge:
    """Serializable graph edge row.

    Attributes:
        source: Source node identifier.
        target: Target node identifier.
        edge_type: Semantic relation type.
        weight: Numeric edge weight.
        article_id: Article provenance identifier.
        event_id: Event provenance identifier.
        attributes_json: JSON string with extra edge attributes.
    """

    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    article_id: str = ""
    event_id: str = ""
    attributes_json: str = "{}"
