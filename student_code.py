"""
Implementation of TraversableDigraph and DAG classes.
Extends SortableDigraph with DFS, BFS, and cycle-safe edge addition.
"""

from collections import deque


class SortableDigraph:
    """Base directed graph supporting topological sorting."""

    def __init__(self):
        self.adj = {}              # adjacency list: {node: {neighbor: weight}}
        self.node_values = {}      # optional node values

    def add_node(self, node, value=None):
        """Add a node with optional value."""
        if node not in self.adj:
            self.adj[node] = {}
        if value is not None:
            self.node_values[node] = value

    def add_edge(self, start, end, edge_weight=1):
        """Add a directed edge with optional weight."""
        if start not in self.adj:
            self.add_node(start)
        if end not in self.adj:
            self.add_node(end)
        self.adj[start][end] = edge_weight

    def get_nodes(self):
        """Return list of all nodes."""
        return list(self.adj.keys())

    def get_node_value(self, node):
        """Return stored node value."""
        return self.node_values.get(node)

    def get_edge_weight(self, start, end):
        """Return the weight of a given edge."""
        return self.adj[start][end]

    def successors(self, node):
        """Return direct successors of node."""
        return list(self.adj.get(node, {}).keys())

    def predecessors(self, node):
        """Return direct predecessors of node."""
        preds = []
        for u, nbrs in self.adj.items():
            if node in nbrs:
                preds.append(u)
        return preds

    def top_sort(self):
        """Perform Kahn’s algorithm for topological sorting."""
        indegree = {u: 0 for u in self.adj}
        for u in self.adj:
            for v in self.adj[u]:
                indegree[v] = indegree.get(v, 0) + 1

        queue = deque([u for u, d in indegree.items() if d == 0])
        order = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.adj.get(u, {}):
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.adj):
            raise ValueError("Graph contains a cycle; topological sort invalid.")
        return order


class TraversableDigraph(SortableDigraph):
    """Directed graph with DFS and BFS traversal."""

    def dfs(self, start):
        """Yield nodes reachable from start using depth-first search."""
        visited = set()
        stack = [start]
        visited.add(start)
        stack_path = []

        while stack:
            node = stack.pop()
            if node not in stack_path and node != start:
                yield node
            stack_path.append(node)
            for neighbor in self.adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    def bfs(self, start):
        """Yield nodes reachable from start using breadth-first search."""
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in self.adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    yield neighbor


class DAG(TraversableDigraph):
    """Directed Acyclic Graph (DAG) preventing cycles on edge addition."""

    def add_edge(self, start, end, edge_weight=1):
        """Add edge if it doesn’t create a cycle."""
        if self._reachable(end, start):
            raise ValueError(f"Adding edge {start} → {end} would create a cycle.")
        super().add_edge(start, end, edge_weight)

    def _reachable(self, start, target):
        """Return True if target reachable from start."""
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            for neighbor in self.adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False
