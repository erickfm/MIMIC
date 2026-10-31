"""Give each agent a fixed costume in vs_phillip.py replays, so ports alternating doesn't swap colors.

Costume is cosmetic: one byte per player in the .slp Game Start event (offset 0x68 + 0x24*port from the
event's command byte), read by playback when it sets up the match. Writes recolored copies to
<dir>/replays_colored/; the originals are untouched. Each replay is matched to its games.jsonl row
by order and checked against that row's final stocks.

  python3 tools/recolor_replays.py checkpoints/vs_phillip_20261029/master [--mimic 3 --phillip 2]

Fox costumes: 0 default (white), 1 red, 2 blue, 3 green.
"""
import argparse
import json
from pathlib import Path

import peppi_py

GAME_START = 0x36
EVENT_PAYLOADS = 0x35


def game_start_offset(raw: bytes) -> int:
    """File offset of the Game Start command byte (the raw stream follows a 15-byte UBJSON header)."""
    i = raw.index(b"[$U#l") + 9          # "raw": [$U#l + 4-byte length
    assert raw[i] == EVENT_PAYLOADS
    sizes_len = raw[i + 1]
    j = i + 1 + sizes_len
    assert raw[j] == GAME_START
    return j


def final_stocks(path: Path) -> dict:
    g = peppi_py.read_slippi(str(path))
    out = {}
    for pl, port in zip(g.start.players, g.frames.ports):
        out[int(pl.port.value) + 1] = int(port.leader.post.stocks[-1].as_py())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--mimic", type=int, default=3)
    ap.add_argument("--phillip", type=int, default=2)
    args = ap.parse_args()

    games = [json.loads(l) for l in open(args.dir / "games.jsonl")]
    replays = sorted((args.dir / "replays").glob("*.slp"))[:len(games)]
    out = args.dir / "replays_colored"
    out.mkdir(exist_ok=True)
    for g, p in zip(games, replays):
        opp = 3 - g["mimic_port"]
        stocks = final_stocks(p)
        assert (stocks[g["mimic_port"]], stocks[opp]) == (g["mimic_stocks"], g["phillip_stocks"]), (p, g, stocks)
        raw = bytearray(p.read_bytes())
        gs = game_start_offset(raw)
        for port, costume in ((g["mimic_port"], args.mimic), (opp, args.phillip)):
            raw[gs + 0x68 + 0x24 * (port - 1)] = costume
        (out / p.name).write_bytes(raw)
    print(f"{len(games)} replays -> {out}")


if __name__ == "__main__":
    main()
