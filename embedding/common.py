class Embeddings(object):

    """
    An abstract embedding object.

    All embedding objects must inherit this interface. `name` property
    must be defined first.

    Embedding objects are designed to be hashable, such that the same type of
    embeddings can be loaded only once.
    """
    name = None

    @property
    def dim(self):
        """Returns embedding dimensions."""
        raise NotImplementedError()

    def preload(self):
        """Load actual data here if necessary."""
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __getitem__(self, item):
        """Returns a numpy array of the embedding."""
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class WordEmbeddingManager(object):

    def __init__(self):
        self.embeds = dict()

    def __getitem__(self, item: Embeddings):
        key = hash(item)
        if key not in self.embeds:
            item.preload()
            self.embeds[key] = item
        return self.embeds[key]