class DirectedGraph:
    """
    This class defines the graph topology.
    Adapted from https://gitlab.epfl.ch/sacs/ml-rawdatasharing/dnn-recommender/-/blob/master/api.py
    """

    def __init__(self, n_procs=None):
        """
        Constructor

        Parameters
        ----------
        n_procs : int, optional
            Number of processes in the graph, if already known

        """
        if n_procs is not None:
            self.n_procs = n_procs
            self.adj_list = [set() for i in range(self.n_procs)]

    def get_all_nodes(self):
        return [i for i in range(self.n_procs)]

    def __update_incomming_edges__(self, node, neighbours):
        """Inserts `node` into the adjacency list of all `neighbors`.

        Args:
            node: int
                The vertex in question
            neighbours: list(int)
                A list of neighbours of the `node`
        """
        for neigh in neighbours:
            self.adj_list[neigh].add(node)

    def __update_outgoing_edges__(self, node, neighbours):
        """Inserts `neighbours` into the adjacency list of `node`.

        Args:
            node: int
                The vertex in question
            neighbours: list(int)
                A list of neighbours of the `node`
        """
        self.adj_list[node].update(neighbours)

    def write_graph_to_file(self, file, type="edges"):
        """
        Writes graph to file

        Parameters
        ----------
        file : str
            File path
        type : str
            One of {"edges", "adjacency"}. Writes the corresponding format.

        """
        with open(file, "w") as of:
            of.write(str(self.n_procs) + "\n")
            if type == "edges":
                for node, adj in enumerate(self.adj_list):
                    for neighbor in adj:
                        of.write("{} {}".format(node, neighbor) + "\n")
            elif type == "adjacency":
                for adj in self.adj_list:
                    of.write(str(*adj) + "\n")
            else:
                raise ValueError("type must be from {edges, adjacency}!")

    def outgoing_edges(self, uid):
        """
        Gives the neighbors of a node

        Parameters
        ----------
        uid : int
            globally unique identifier of the node

        Returns
        -------
        set(int)
            a set of neighbours

        """
        return self.adj_list[uid]

    def incomming_edges(self, uid):
        """
        Gives the neighbors of a node

        Parameters
        ----------
        uid : int
            globally unique identifier of the node

        Returns
        -------
        set(int)
            a set of neighbours

        """
        incomming = []
        for i, adj in enumerate(self.adj_list):
            if uid in adj:
                incomming.append(i)
        return
