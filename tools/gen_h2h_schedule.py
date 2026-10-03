"""Generate a broad, connected h2h matchup schedule across all 23 characters.

Not a dense all-play-all (253 pairs); a designed sparse distribution that is
(a) connected and scale-pinned for Bradley-Terry/Elo, (b) locally dense where
discrimination is hardest, (c) seeded with deliberately weird combos.

Pairs = union of, per character in the CPU-9 prior order:
  - local band  : neighbors +-1, +-2 (fine local resolution + a connected chain)
  - anchors     : 4 stratified rungs spanning top->bottom (pins global scale)
  - random      : 2 random opponents (breaks schedule regularities)
plus a curated set of weird/extreme pairings (max info per game).
"""
import random
random.seed(42)

# CPU-9 prior order, best -> worst (data breaks ties). The h2h produces the
# real ranking; this only shapes the schedule.
PRIOR = ["fox","sheik","marth","falco","puff","cptfalcon","doc","dk","peach",
         "samus","ice_climbers","ganondorf","pikachu","bowser","roy","luigi",
         "yoshi","mario","link","gameandwatch","mewtwo","ylink","ness"]
N = len(PRIOR)
idx = {c: i for i, c in enumerate(PRIOR)}
ANCHOR_RANKS = [2, 8, 13, 19]   # marth, peach, pikachu, gameandwatch

WEIRD = [
    ("fox","ness"), ("sheik","ylink"), ("doc","yoshi"), ("dk","luigi"),
    ("bowser","falco"), ("bowser","marth"), ("roy","mario"),
    ("mewtwo","ganondorf"), ("pikachu","samus"), ("ice_climbers","peach"),
    ("gameandwatch","mewtwo"), ("link","ylink"), ("puff","bowser"),
    ("fox","sheik"), ("doc","mario"), ("ganondorf","bowser"),
    ("samus","link"), ("marth","roy"),
]

pairs = set()
def add(a, b):
    if a != b:
        pairs.add(tuple(sorted((a, b))))

for i, c in enumerate(PRIOR):
    for d in (-2, -1, 1, 2):                 # local band
        j = i + d
        if 0 <= j < N:
            add(c, PRIOR[j])
    for r in ANCHOR_RANKS:                    # stratified anchors
        add(c, PRIOR[r])
    others = [x for x in PRIOR if x != c and tuple(sorted((c, x))) not in pairs]
    for x in random.sample(others, min(2, len(others))):   # random spread
        add(c, x)
for a, b in WEIRD:
    add(a, b)

pairs = sorted(pairs)
# opponent-count per char (connectivity check)
deg = {c: 0 for c in PRIOR}
for a, b in pairs:
    deg[a] += 1; deg[b] += 1

if __name__ == "__main__":
    import json, sys
    print(f"{len(pairs)} unique pairings across {N} chars")
    print("degree (opponents) per char:")
    for c in PRIOR:
        print(f"  {c:13s} {deg[c]}")
    print(f"min degree {min(deg.values())}, max {max(deg.values())}, "
          f"mean {sum(deg.values())/N:.1f}")
    json.dump([list(p) for p in pairs],
              open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"))
