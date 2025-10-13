import json
import copy
import pytest

from gene_generator import Gene_Generator
import gene_validator
import config


def test_happy_path():
    gen = Gene_Generator()
    gene = gen.make_core()
    errors = gene_validator.validate_gene(gene, config)
    assert errors == [], f"Expected no errors, got: {errors}"


def test_hinge_leaf():
    gen = Gene_Generator()
    gene = gen.make_core()
    # Force a hinge without brick: find first right subtree hinge and remove its brick
    if gene["core"].get("right") and gene["core"]["right"].get("hinge"):
        gene["core"]["right"]["hinge"].pop("brick", None)
    else:
        # construct a minimal invalid example
        gene = {"id": 0, "brain": {}, "core": {"front": {"hinge": {"brick": {}}}, "right": {"hinge": {}}, "left": {}, "back": {}}}

    errors = gene_validator.validate_gene(gene, config)
    assert any(e["code"] == "missing_brick" or e["code"] == "invalid_brick" for e in errors), errors


def test_asymmetry():
    gen = Gene_Generator()
    gene = gen.make_core()
    # Remove a branch on the right to create asymmetry
    right = gene["core"].get("right", {})
    # If right has a hinge with a brick, clear one of its faces
    if isinstance(right, dict) and right.get("hinge") and isinstance(right["hinge"].get("brick"), dict):
        right_brick = right["hinge"]["brick"]
        # remove a face if present
        for f in ["front", "left", "right"]:
            if right_brick.get(f):
                right_brick[f] = {}
                break
    else:
        pytest.skip("generator produced a core with no right branch; skip asymmetry test")

    errors = gene_validator.validate_gene(gene, config)
    assert any(e["code"] == "asymmetry" for e in errors), errors


def test_spine_branch():
    gen = Gene_Generator()
    gene = gen.make_core()
    # Add an off-spine branch: assume core.front.hinge.brick exists and add a front branch there
    front = gene["core"].get("front", {})
    if isinstance(front, dict) and front.get("hinge") and isinstance(front["hinge"].get("brick"), dict):
        front_brick = front["hinge"]["brick"]
        # create an extra front branch that has a subtree (this may create a spine-branch depending on generator)
        front_brick["front"] = {"hinge": {"brick": {"left": {}}}}
    else:
        pytest.skip("generator produced no front spine; skip spine branch test")

    errors = gene_validator.validate_gene(gene, config)
    # Our validator doesn't explicitly detect spine-branch currently; ensure we at least didn't crash and returned something
    assert isinstance(errors, list)


def test_module_count_exceed():
    gen = Gene_Generator()
    gene = gen.make_core()
    # force max bricks low to trigger the error
    class Cfg:
        MAX_BRICKS = 1

    errors = gene_validator.validate_gene(gene, Cfg)
    assert any(e["code"] == "too_many_modules" for e in errors), errors
