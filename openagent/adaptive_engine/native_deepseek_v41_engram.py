"""Private, explicit-library Engram ABI. All numerical execution is native C."""
import ctypes as C
from itertools import islice
import operator
from pathlib import Path
import threading
import weakref
import _ctypes
import os


def release(handle):
    if os.name == "nt":
        _ctypes.FreeLibrary(handle)
    else:
        _ctypes.dlclose(handle)


class Head(C.Structure):
    _fields_ = [("order", C.c_uint32), ("width", C.c_uint32),
                ("rows", C.c_uint32), ("reserved", C.c_uint32),
                ("seed", C.c_uint64), ("multiplier", C.c_uint64 * 4),
                ("table", C.POINTER(C.c_float)), ("count", C.c_size_t)]


class Config(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in
                ("abi", "heads", "dim", "branches", "vocabulary", "pad")]
    _fields_ += [("epsilon", C.c_float), ("reserved", C.c_uint32),
                 ("head", C.POINTER(Head))]
    _fields_ += [(n, C.POINTER(C.c_float)) for n in
                 ("key", "value", "query_scale", "key_scale")]
    _fields_ += [(n, C.c_size_t) for n in
                 ("key_count", "value_count", "query_count", "scale_count", "weight_budget")]


class State(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("abi", "length", "capacity", "reserved")]
    _fields_ += [("tokens", C.c_uint32 * 4096)]


def integer(value, low, high):
    if isinstance(value, bool):
        raise ValueError("boolean is not an ID or dimension")
    value = operator.index(value)
    if not low <= value <= high:
        raise ValueError("integer outside bounded contract")
    return value


def array(values, count, kind=C.c_float):
    values = tuple(islice(iter(values), count + 1))
    if len(values) != count:
        raise ValueError("element count does not match geometry")
    return (kind * count)(*values)


def check(status):
    if status:
        raise ValueError({1: "invalid geometry, state, IDs or capacity", 2: "admission budget exceeded",
                          3: "nonfinite input or intermediate"}.get(status, "native failure"))


class EngramSession:
    """Append-only compressed-token history with destructive prefix rollback."""
    def __init__(self, owner, capacity):
        self._owner = owner
        self._state = State()
        check(owner.library.oa_v41_engram_state_init(C.byref(self._state), capacity))

    @property
    def length(self):
        with self._owner._lock:
            return self._state.length

    def rollback(self, length):
        with self._owner._lock:
            self._owner._check_open()
            length = integer(length, 0, self._state.length)
            self._owner._check_open()
            check(self._owner.library.oa_v41_engram_rollback(
                C.byref(self._state), length))

    def forward(self, compressed_ids, hidden, *, budget=8 * 1024 * 1024):
        with self._owner._lock:
            model = self._owner
            model._check_open()
            ids = tuple(islice(iter(compressed_ids), 257))
            if len(ids) > 256:
                raise ValueError("chunk exceeds 256 tokens")
            tokens = array((integer(v, 0, model._config.vocabulary - 1) for v in ids),
                           len(ids), C.c_uint32)
            size = len(ids) * model._config.branches * model._config.dim
            x = array(hidden, size)
            y = (C.c_float * size)()
            addresses = (C.c_uint32 * (len(ids) * model._config.heads))()
            budget = integer(budget, 0, C.c_size_t(-1).value)
            model._check_open()
            check(model.library.oa_v41_engram_run(C.byref(model._config), C.byref(self._state),
                tokens, len(ids), x, size, y, size, addresses, len(addresses),
                budget))
            return tuple(y), tuple(addresses)


class NativeDeepSeekV41Engram:
    """Owns immutable copies of supplied FP32 weights; no implicit hash defaults.

    Each head is a mapping with order, width, rows, seed, multipliers (four
    uint64s), and table (row-major). Head-list order defines concatenation.
    Key: [branches, dim, memory]; value: [dim, memory]; scales: [branches, dim].
    """
    def __init__(self, library_path, *, heads, dim, branches, vocabulary, pad,
                 key, value, query_scale, key_scale, epsilon, weight_budget):
        self._lock = threading.RLock()
        self._closed = False
        path = Path(library_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("explicit trusted library must be a regular file")
        self.library = C.CDLL(str(path))
        self._finalizer = weakref.finalize(self, release, self.library._handle)
        lib = self.library
        lib.oa_v41_engram_abi_version.argtypes = []
        lib.oa_v41_engram_abi_version.restype = C.c_uint32
        if lib.oa_v41_engram_abi_version() != 1:
            raise ValueError("unsupported Engram ABI")
        lib.oa_v41_engram_state_init.argtypes = [C.POINTER(State), C.c_uint32]
        lib.oa_v41_engram_rollback.argtypes = [C.POINTER(State), C.c_uint32]
        lib.oa_v41_engram_run.argtypes = [C.POINTER(Config), C.POINTER(State),
            C.POINTER(C.c_uint32), C.c_size_t, C.POINTER(C.c_float), C.c_size_t,
            C.POINTER(C.c_float), C.c_size_t, C.POINTER(C.c_uint32), C.c_size_t, C.c_size_t]
        for name in ("state_init", "rollback", "run"):
            getattr(lib, "oa_v41_engram_" + name).restype = C.c_int
        dim, branches = integer(dim, 1, 256), integer(branches, 1, 8)
        vocabulary = integer(vocabulary, 1, 2**32 - 1)
        pad = integer(pad, 0, vocabulary - 1)
        weight_budget = integer(weight_budget, 0, 2**30)
        heads = tuple(islice(iter(heads), 25))
        if not 1 <= len(heads) <= 24:
            raise ValueError("head count outside bounds")
        descriptors = []
        memory, total = 0, 0
        for head in heads:
            order = integer(head["order"], 2, 4)
            width, rows = integer(head["width"], 1, 256), integer(head["rows"], 1, 33554432)
            memory += width
            total += width * rows
            descriptors.append((head, order, width, rows))
        if {a[1] for a in descriptors} != {2, 3, 4}:
            raise ValueError("all orders 2/3/4 are required")
        total += (branches + 1) * dim * memory + 2 * branches * dim
        if total * 4 > weight_budget:
            raise ValueError("weight admission budget exceeded before allocation")
        self._storage = []
        native_heads = (Head * len(heads))()
        for i, (head, order, width, rows) in enumerate(descriptors):
            table = array(head["table"], rows * width)
            multipliers = array((integer(v, 0, 2**64 - 1) for v in head["multipliers"]), 4, C.c_uint64)
            native_heads[i] = Head(order, width, rows, 0, integer(head["seed"], 0, 2**64 - 1),
                                   multipliers, table, len(table))
            self._storage.append(table)
        k, v = array(key, branches * dim * memory), array(value, dim * memory)
        q, ks = array(query_scale, branches * dim), array(key_scale, branches * dim)
        self._storage.extend((native_heads, k, v, q, ks))
        self._config = Config(1, len(heads), dim, branches, vocabulary, pad, epsilon, 0,
                              native_heads, k, v, q, ks, len(k), len(v), len(q), len(ks), weight_budget)
        # Empty execution validates native configuration without numerical Python work.
        self.session().forward([], [])

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Engram is closed")

    def close(self):
        with self._lock:
            self._closed = True
            self._finalizer()

    def session(self, capacity=4096):
        with self._lock:
            self._check_open()
            capacity = integer(capacity, 1, 4096)
            self._check_open()
            return EngramSession(self, capacity)

    def forward(self, compressed_ids, hidden, *, budget=8 * 1024 * 1024):
        """Independent full sequence (up to 256); returns deltas and hash rows."""
        return self.session().forward(compressed_ids, hidden, budget=budget)
