# GameCube Controller Input Resolution & Label Collapse

How 31 possible button combinations reduce to 7 distinct behaviors, and how to use that for single-label classification.

---

## 1. Setup

Five input classes:

| Class | Physical button(s) |
|---|---|
| A | A |
| B | B |
| Z | Z |
| JUMP | X or Y |
| TRIG | L or R |

When multiple buttons are pressed on the same frame, the game resolves the input to a single behavior. Almost always the combination collapses to one of its constituent buttons — A + B produces the same behavior as B alone, so A + B is an *alias* for B. In exactly one case, a combination produces *emergent* behavior that no single button produces: A + TRIG.

This document records the results of an exhaustive sweep over all 31 non-empty combinations, derives the resolution rules, and applies them to collapse a multi-label input problem into a 7-class multiclass problem suitable for machine learning.

---

## 2. The Sweep

Every unordered combination of the five classes was tested, from 1 button up to all 5 pressed simultaneously. That's C(5,1) + C(5,2) + C(5,3) + C(5,4) + C(5,5) = 5 + 10 + 10 + 5 + 1 = **31** combinations, plus the empty state handled as "none."

### Two buttons

| Input | Resolves to |
|---|---|
| A + B | B |
| A + Z | Z |
| A + JUMP | A |
| A + TRIG | **A + TRIG** |
| B + Z | B |
| B + JUMP | B |
| B + TRIG | B |
| Z + JUMP | Z |
| Z + TRIG | Z |
| JUMP + TRIG | TRIG |

### Three buttons

| Input | Resolves to |
|---|---|
| A + B + Z | B |
| A + B + JUMP | B |
| A + B + TRIG | B |
| A + Z + JUMP | Z |
| A + Z + TRIG | **A + TRIG** |
| A + JUMP + TRIG | **A + TRIG** |
| B + Z + JUMP | B |
| B + Z + TRIG | B |
| B + JUMP + TRIG | B |
| Z + JUMP + TRIG | Z |

### Four buttons

| Input | Resolves to |
|---|---|
| A + B + Z + JUMP | B |
| A + B + Z + TRIG | B |
| A + B + JUMP + TRIG | B |
| A + Z + JUMP + TRIG | **A + TRIG** |
| B + Z + JUMP + TRIG | B |

### Five buttons

A + B + Z + JUMP + TRIG resolves to **B**.

---

## 3. The Resolution Rules

All 31 results are described by three rules, applied in order:

1. **If B is pressed → B.** B is a hard override. It beats everything, including the A + TRIG combo.
2. **Else if A and TRIG are both pressed → A + TRIG.** This pair is protected against Z and JUMP; only B can break it.
3. **Else highest priority wins:** **Z > A ≈ TRIG > JUMP.** A and TRIG share a tier but never collide here — any input containing both would have been caught by Rule 2.

Rule order matters. Checking Rule 2 before Rule 1 would mislabel B + A + TRIG as A + TRIG when the true behavior is B.

### Per-class behavior

- **B** — hard override. Wins against everything.
- **Z** — wins all priority contests but can't break A + TRIG.
- **A** — loses to Z alone, protected when paired with TRIG.
- **TRIG** — loses to Z alone, protected when paired with A.
- **JUMP** — inert. Never wins, never pairs. Only produces output when pressed alone.

The 5 classes can't be reduced further. A and TRIG share a priority tier but can't merge because they produce distinct solo behaviors *and* form A + TRIG with each other. JUMP is inert in combos but still has a unique solo behavior.

---

## 4. Collapsing to a Single Label

### Why collapse

Treating this as a multi-label problem means predicting 5 independent binary flags, which is a joint distribution over 2^5 = 32 states. But the game only exposes 7 distinct behaviors, so 25 of those 32 states are aliases for simpler ones. A multi-label model would waste capacity learning redundant structure that the game itself doesn't expose.

A multiclass framing over the 7 behaviors is lossless with respect to in-game behavior and dramatically simpler to learn. The 7 classes are: A, B, Z, JUMP, TRIG, A + TRIG, none.

The resolution rules define an equivalence relation over the 32-state input space — two inputs are equivalent if they produce the same behavior — partitioning it into 7 classes. Relabeling is just applying the rules to each row.

Note that collapsing is lossy with respect to raw button state: after relabeling, B and B + A + TRIG both become "B" and can't be distinguished. This is usually what you want for behavior modeling; flag it if you ever need to recover actual input state.

### Relabeling function

```python
def collapse_input(a: bool, b: bool, z: bool, jump: bool, trig: bool) -> str:
    if b:
        return "B"
    if a and trig:
        return "A+TRIG"
    if z:
        return "Z"
    if a:
        return "A"
    if trig:
        return "TRIG"
    if jump:
        return "JUMP"
    return "none"
```

### Full label map

| Input | Label |
|---|---|
| (none) | none |
| A | A |
| B | B |
| Z | Z |
| JUMP | JUMP |
| TRIG | TRIG |
| A + B | B |
| A + Z | Z |
| A + JUMP | A |
| A + TRIG | A+TRIG |
| B + Z | B |
| B + JUMP | B |
| B + TRIG | B |
| Z + JUMP | Z |
| Z + TRIG | Z |
| JUMP + TRIG | TRIG |
| A + B + Z | B |
| A + B + JUMP | B |
| A + B + TRIG | B |
| A + Z + JUMP | Z |
| A + Z + TRIG | A+TRIG |
| A + JUMP + TRIG | A+TRIG |
| B + Z + JUMP | B |
| B + Z + TRIG | B |
| B + JUMP + TRIG | B |
| Z + JUMP + TRIG | Z |
| A + B + Z + JUMP | B |
| A + B + Z + TRIG | B |
| A + B + JUMP + TRIG | B |
| A + Z + JUMP + TRIG | A+TRIG |
| B + Z + JUMP + TRIG | B |
| A + B + Z + JUMP + TRIG | B |

---

## 5. Implementation Notes

**Pipeline.** Apply `collapse_input` row-by-row to produce a single label per frame, then train a standard multiclass classifier on the collapsed targets.

**Class imbalance.** The collapse will produce a skewed distribution. Expect "none" to dominate, B and Z to be common (they absorb many compound inputs), and A + TRIG to be rare (requires two specific buttons and no B). Measure before training and apply standard remedies: class weighting, resampling, or focal loss.

**Scope and caveats.** This analysis covers single-frame digital-button inputs only. Not tested, and may warrant investigation if relevant:

- **Held vs newly-pressed.** Holding B and tapping A mid-hold may differ from pressing both on the same frame.
- **Sub-class splits.** JUMP is X or Y, TRIG is L or R. Whether X alone, Y alone, and X + Y all behave identically (same for L/R) was not verified.
- **Cross-class sub-combos.** X + L was not tested independently from the full JUMP + TRIG.
- **Analog inputs.** Stick, C-stick, and analog trigger pressure are out of scope.
- **Polling order.** Whether same-frame resolution is by priority or by poll order is indistinguishable from the data gathered here.

Within the stated scope, the model is exhaustively verified: every one of the 31 combinations was tested and every result is predicted by the three rules.
