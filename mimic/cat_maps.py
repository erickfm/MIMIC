"""
Mapping Melee enums → dense embedding indices.
We map each enum's `.value` (raw game ID) to a contiguous 0..N-1 index (suitable for embedding lookups).
"""
from melee.enums import Stage, Character, Action, ProjectileType

# Dense 0..N-1 index maps for embedding, keyed by raw game ID (.value)
STAGE_MAP = {s.value: i for i, s in enumerate(Stage)}
CHARACTER_MAP = {c.value: i for i, c in enumerate(Character)}
ACTION_MAP = {a.value: i for i, a in enumerate(Action)}
PROJECTILE_TYPE_MAP = {p.value: i for i, p in enumerate(ProjectileType)}

# ---------------------------------------------------------------------------
# HAL-specific categorical maps (6 stages, 27 characters, 396 actions)
# ---------------------------------------------------------------------------
HAL_STAGES = [
    "FINAL_DESTINATION", "BATTLEFIELD", "POKEMON_STADIUM",
    "DREAMLAND", "FOUNTAIN_OF_DREAMS", "YOSHIS_STORY",
]
HAL_STAGE_MAP = {s.value: i for i, s in enumerate(
    s for s in Stage if s.name in HAL_STAGES)}

HAL_CHARACTERS = [
    "MARIO", "FOX", "CPTFALCON", "DK", "KIRBY", "BOWSER", "LINK", "SHEIK",
    "NESS", "PEACH", "POPO", "NANA", "PIKACHU", "SAMUS", "YOSHI",
    "JIGGLYPUFF", "MEWTWO", "LUIGI", "MARTH", "ZELDA", "YLINK", "DOC",
    "FALCO", "PICHU", "GAMEANDWATCH", "GANONDORF", "ROY",
]
HAL_CHARACTER_MAP = {c.value: i for i, c in enumerate(
    c for c in Character if c.name in HAL_CHARACTERS)}

# HAL uses all Action enum values; checkpoint has Embedding(396, 32)
HAL_ACTION_MAP = {a.value: i for i, a in enumerate(Action)}
