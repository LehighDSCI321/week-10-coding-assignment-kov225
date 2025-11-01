"""Graph module implementing TraversableDigraph and DAG classes.

"""

from collections import deque


class SortableDigraph:
    """Base directed graph supporting topological sorting."""

    def __init__(self):
        """Initialize adjacency and node-value dictionaries."""
        self.adj = {}          # {node: {neighbor: weight}}
        self.node_values = {}  # {node: value}

    def add_node(self, node, value=None):
        """Add a node to the graph with an optional value."""
        if node not in self.adj:
            self.adj[node] = {}
        if value is not None:
            self.node_values[node] = value

    def add_edge(self, start, end, edge_weight=1):
        """Add a directed edge between two nodes with a weight."""
        if start not in self.adj:
            self.add_node(start)
        if end not in self.adj:
            self.add_node(end)
        self.adj[start][end] = edge_weight

    def get_nodes(self):
        """Return a list of all nodes."""
        return list(self.adj.keys())

    def get_node_value(self, node):
        """Return the value assigned to a node."""
        return self.node_values.get(node)

    def get_edge_weight(self, start, end):
        """Return the weight of a given edge."""
        return self.adj[start][end]

    def successors(self, node):
        """Return all immediate successors of a node."""
        return list(self.adj.get(node, {}).keys())

    def predecessors(self, node):
        """Return all immediate predecessors of a node."""
        preds = []
        for src, nbrs in self.adj.items():
            if node in nbrs:
                preds.append(src)
        return preds

    def top_sort(self):
        """Perform topological sort using Kahn’s algorithm."""
        indegree = {node: 0 for node in self.adj}
        for src in self.adj:
            for dst in self.adj[src]:
                indegree[dst] = indegree.get(dst, 0) + 1

        queue = deque([n for n, deg in indegree.items() if deg == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for nbr in self.adj.get(node, {}):
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    queue.append(nbr)

        if len(order) != len(self.adj):
            raise ValueError("Graph contains a cycle; topological sort invalid.")
        return order


class TraversableDigraph(SortableDigraph):
    """Directed graph that supports DFS and BFS traversal."""

    def dfs(self, start):
        """Yield nodes reachable from start using depth-first search."""
        visited = set()
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for neighbor in self.adj.get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    yield neighbor
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
    """Directed acyclic graph enforcing cycle prevention on edge addition."""

    def add_edge(self, start, end, edge_weight=1):
        """Add a directed edge if it does not create a cycle."""
        if self._reachable(end, start):
            raise ValueError(f"Adding edge {start} → {end} would create a cycle.")
        super().add_edge(start, end, edge_weight)

    def _reachable(self, start, target):
        """Return True if target is reachable from start."""
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
