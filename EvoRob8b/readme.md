# Overview



## Table of contents
- [Overview](#overview)
  - [Table of contents](#table-of-contents)
  - [Group \& Project](#group--project)
  - [Codebase](#codebase)
  - [To do](#to-do)
  - [Morphology rules](#morphology-rules)
    - [Quick overview](#quick-overview)
    - [JSON gene layout](#json-gene-layout)
    - [Spine and symmetry (plain language)](#spine-and-symmetry-plain-language)
    - [Constraints the generator enforces](#constraints-the-generator-enforces)
    - [Constraints imposed during EA](#constraints-imposed-during-ea)
    - [Validator checklist (actionable)](#validator-checklist-actionable)
    - [Suggested validator tests (practical)](#suggested-validator-tests-practical)
    - [Implementation hints](#implementation-hints)
    - [Summary](#summary)


## Group & Project

- Group: __[Brains and brawns - Evorob 3D]__
- Institution: __[University of Oslo]__
- Project title: __[Comparing a EA generated body to a human made body]__
- Project lead: __[Kyrre ]__ <__[kyrrehg@ifi.uio.no]__>
- Team members: __[Emma F. Brovold], [Ka Thas], [Sofie L. Markeset], [Vebjørn B. Karlsen]__

Short description:
__[We limiting the search space of an ea while comparing its outputs to human benchmarks]__

Primary goals:
- __[Compare a EA generated morphology to one deliberately created by humans]__
- __[Learning multiple tasks sequentially vs. simultaneously in parallell]__

Start date: __[2025-09-16]__
License: __[GNU LESSER GENERAL PUBLIC LICENSE]__
Repository: __[[REPO_URL](https://github.com/ka-thas/revolve2)]__

Contact & contribution:
- Primary contact: __[Ka Thas]__ <__[kavint@ifi.uio.no]__>
<!-- - Contribution guidelines: __[path/to/CONTRIBUTING.md or short instructions]__  -->


## Codebase

- [gene_generator](./gene_generator.py)
- [parse_gene](./parse_gene.py)
- [EA](./EA.py)
  - comprehensive evolutionary algorithm for JSON gene representation


## To do
- [x] Gene generator
 - [x] BFS generation
   - for å generere lemmene
 - [x] max parts 
 - [x] symmetry 
 - [x] orientation
   - [x] revisit symmetry
- [x] higher prob of placing bricks on front than on right and left


- [ ] [Gene validator](./gene_validator.py)
  - [ ] brick has front, right, left faces
    - [ ] especially spine
  - [ ] hinge has brick
  - [ ] spine symmetry
  - [ ] module count

- [ ] brain representation

- [ ] EA
 - [ ] Crossover !!!
 - [x] Mutation
 - [ ] Inner learning loop for brain optimization @brains
   - [x] Mutation
   - [x] Eval
   - [x] Selection
 - [x] Evaluation
 - [x] Tournament selection

## Morphology rules

morphology rules applied to genes produced by the generator.

### Quick overview

- Bricks are always attached via hinges. Bricks never appear as standalone
  leaf nodes.
- Hinges cannot be leaves: every hinge must connect to a brick on its output
  face.
- The body must be left/right symmetric around a central spine. The generator
  builds the right side and mirrors it to the left.
- Module ordering follows the pattern: brick -> face -> hinge -> brick -> ...

### JSON gene layout

Top-level keys:
- `id`: identifier (int or string)
- `brain`: brain-specific data object (may be empty during morphology-only
  evolution)
- `core`: the root brick description with faces `front`, `right`, `left`,
  `back`

Faces are objects keyed by face names. A non-empty face contains a `hinge`
object, and a `hinge` must contain a `brick` object. Example:

```json
{
  "id": 1,
  "brain": {},
  "core": {
    "front": { "hinge": { "brick": {} } },
    "right": {},
    "left": {},
    "back": {}
  }
}
```

Which faces are valid where:
- Core faces: `front`, `right`, `left`, `back`
- Brick faces: `front`, `right`, `left` (bricks do not expose a `back` face)
- Hinge face: `front` (hinge attaches forward to a brick)

### Spine and symmetry (plain language)

- The spine is the central chain of bricks you get by following only `front`
  and `back` faces starting at the `core`. The spine should be a straight
  sequence (no branching along the spine direction).
- All limbs on the right subtree must have structural mirrors on the left
  subtree (topology should be symmetric). Orientation may change according to
  implementation details, but topology must mirror.

### Constraints the generator enforces

- Maximum bricks/modules (configurable via `config.MAX_BRICKS`) — the
  generator prevents constructions that exceed this number.
- No hinge may be terminal (leaf) — every hinge must lead to a brick.
- Empty faces must be empty objects (`{}`), not `null` or omitted.
- The generator assigns a default `orientation` field where appropriate.

### Constraints imposed during EA

- Brick follows all hinges
- Symmetry
- 

### Validator checklist (actionable)

1. Gene is a JSON object with `id`, `brain`, `core` keys.
2. `core` contains exactly `front`, `right`, `left`, `back` faces; each face
   is `{}` or a `hinge` object.
3. Every `hinge` contains a `brick` object.
4. No hinge is a leaf (its `brick` must be non-empty).
5. Brick objects only contain `front`, `right`, `left` faces.
6. The spine (follow `front`/`back` from `core`) is a straight chain — no
   branching along that path.
7. Left/right subtree topology is mirrored (structure-wise) between right and
   left.
8. Total module count ≤ configured maximum.

### Suggested validator tests (practical)

- Happy path: `Gene_Generator().make_core()` should pass all checks.
- Hinge leaf test: create a hinge without a brick → must fail.
- Asymmetry test: remove a branch from right subtree → validator reports
  asymmetry.
- Spine-branch test: add an off-spine branch along the spine → validator
  reports spine branching.

### Implementation hints

- Use a deterministic traversal (path strings or canonical coordinates) to
  compare mirrored subtrees.
- Report errors with a short path (e.g. `core.front.hinge.brick.front`) to
  speed debugging.
- Keep checks recursive and local — the gene is a tree, validation is O(n).

### Summary

- Genes are tree-shaped JSON objects with precise face/hinge/brick rules.
- The spine is the central front/back chain and must be straight (no
  branching); the body must be left/right symmetric around that spine.
- Followed the validator checklist when implementing `gene_validator.py`.
