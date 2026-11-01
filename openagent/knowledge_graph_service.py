"""G25: Neo4j-style knowledge graph building and reasoning."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class Node:
    """Knowledge graph node."""
    def __init__(self, node_id: str, label: str, properties: dict[str, Any]):
        self.id = node_id
        self.label = label
        self.properties = properties

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "properties": self.properties}


class Edge:
    """Knowledge graph edge/relationship."""
    def __init__(self, edge_id: str, source: str, target: str, rel_type: str, properties: dict[str, Any]):
        self.id = edge_id
        self.source = source
        self.target = target
        self.type = rel_type
        self.properties = properties

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "properties": self.properties,
        }


class KnowledgeGraphService:
    """Neo4j-style knowledge graph for reasoning and semantic understanding."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.indexes = {}
        self.node_metadata: dict[str, dict[str, Any]] = {}
        self.edge_metadata: dict[str, dict[str, Any]] = {}

    def create_node(self, label: str, properties: dict[str, Any], project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Create knowledge graph node."""
        node_id = str(uuid4())
        node = Node(node_id, label, properties)
        self.nodes[node_id] = node
        self.node_metadata[node_id] = {"project": project, "owner": owner}
        logger.info(f"Created node: {label} ({node_id}) (project={project}, owner={owner})")
        result = node.to_dict()
        result["project"] = project
        result["owner"] = owner
        return result

    def create_edge(self, source_id: str, target_id: str, rel_type: str, properties: dict[str, Any] | None = None, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Create relationship between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return {"error": "Source or target node not found"}

        # Verify source and target belong to same project/owner
        source_meta = self.node_metadata.get(source_id, {})
        target_meta = self.node_metadata.get(target_id, {})
        if (source_meta.get("project") != project or target_meta.get("project") != project
                or source_meta.get("owner") != owner or target_meta.get("owner") != owner):
            return {"error": "Source and target nodes must belong to this project and owner"}

        edge_id = str(uuid4())
        edge = Edge(edge_id, source_id, target_id, rel_type, properties or {})
        self.edges[edge_id] = edge
        self.edge_metadata[edge_id] = {"project": project, "owner": owner}
        logger.info(f"Created edge: {rel_type} ({source_id} -> {target_id}) (project={project}, owner={owner})")
        result = edge.to_dict()
        result["project"] = project
        result["owner"] = owner
        return result

    def find_node(self, label: str | None = None, **properties) -> list[Node]:
        """Find nodes by label and properties."""
        results = []
        for node in self.nodes.values():
            if label and node.label != label:
                continue
            if all(node.properties.get(k) == v for k, v in properties.items()):
                results.append(node)
        return results

    def find_path(self, source_id: str, target_id: str, max_depth: int = 3,
                  project: str | None = None, owner: str | None = None) -> list[list[str]]:
        """Find paths between nodes (BFS).

        When project and owner are given the walk only crosses edges belonging
        to them, so a path cannot be assembled out of another owner's edges.
        """
        from collections import deque

        def allowed(edge_id: str) -> bool:
            if project is None and owner is None:
                return True
            meta = self.edge_metadata.get(edge_id, {})
            return meta.get("project") == project and meta.get("owner") == owner

        visited = set()
        queue = deque([(source_id, [source_id])])
        paths = []

        while queue and len(paths) < 5:
            current, path = queue.popleft()

            if current == target_id:
                paths.append(path)
                continue

            if len(path) >= max_depth or current in visited:
                continue

            visited.add(current)

            # Find connected nodes
            for edge_id, edge in self.edges.items():
                if edge.source == current and allowed(edge_id):
                    next_node = edge.target
                    if next_node not in visited:
                        queue.append((next_node, path + [next_node]))
        
        return paths

    def find_paths(self, start_label: str, end_label: str, project: str = "default", owner: str = "default") -> list[list[dict[str, Any]]]:
        """Alias: Find paths between node labels (for route compatibility)."""
        def mine(node) -> bool:
            meta = self.node_metadata.get(node.id, {})
            return meta.get("project") == project and meta.get("owner") == owner

        start_nodes = [n for n in self.nodes.values() if n.label == start_label and mine(n)]
        end_nodes = [n for n in self.nodes.values() if n.label == end_label and mine(n)]

        all_paths = []
        for start in start_nodes:
            for end in end_nodes:
                paths = self.find_path(start.id, end.id, project=project, owner=owner)
                for path in paths:
                    all_paths.append([self.nodes[nid].to_dict() for nid in path if nid in self.nodes])
        return all_paths

    def get_node_context(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        """Get node and its relationships."""
        if node_id not in self.nodes:
            return {"error": "Node not found"}
        
        node = self.nodes[node_id]
        connected_edges = []
        
        for edge in self.edges.values():
            if edge.source == node_id or edge.target == node_id:
                connected_edges.append(edge.to_dict())
        
        return {
            "node": node.to_dict(),
            "connected_edges": connected_edges,
            "connection_count": len(connected_edges),
        }

    def query_graph(self, query: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Execute a simple graph query (Neo4j-like Cypher subset)."""
        # Simple pattern: find nodes matching keywords
        keywords = query.lower().split()
        matching_nodes = []

        for node_id, node in self.nodes.items():
            node_meta = self.node_metadata.get(node_id, {})
            if node_meta.get("project") != project or node_meta.get("owner") != owner:
                continue
            if any(kw in node.label.lower() for kw in keywords):
                result = node.to_dict()
                result["project"] = project
                result["owner"] = owner
                matching_nodes.append(result)

        return {
            "query": query,
            "results": matching_nodes,
            "result_count": len(matching_nodes),
        }

    def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_labels": list(set(n.label for n in self.nodes.values())),
            "relationship_types": list(set(e.type for e in self.edges.values())),
        }

    def export_graph(self) -> dict[str, Any]:
        """Export complete graph."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "stats": self.get_graph_stats(),
        }
