import enum

COL_MOLECULE_ID = "molecule-ID"


class AtomGroup(enum.Enum):
    ROOT = 1
    FREE = 2
    LEAF = 3
