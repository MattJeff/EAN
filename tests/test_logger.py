"""Smoke-test UnifiedLogger outputs both CSV and JSON lines."""
from pathlib import Path

from hx_monitoring import UnifiedLogger
from hx_core.network import Network
from hx_core.neuron import Neuron
from hx_core.assembly import Assembly


def _dummy_network():
    n = Network()
    asm = Assembly("A", [Neuron([0, 0, 0]) for _ in range(4)])
    n.add_assembly(asm)
    return n


def test_unified_logger(tmp_path: Path):
    csv_path = tmp_path / "log.csv"
    json_path = tmp_path / "log.jsonl"
    logger = UnifiedLogger(csv_path, json_path=json_path)

    net = _dummy_network()
    logger.log(net, reward=0.5)

    # Files created
    assert csv_path.exists()
    assert json_path.exists()

    # Check CSV has header plus one data row
    assert csv_path.read_text().strip().count("\n") == 1
    # JSONL has one line
    assert len(json_path.read_text().splitlines()) == 1
