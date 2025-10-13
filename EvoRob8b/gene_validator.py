"""Gene validator for EvoRob8b genes.

Provides a single entrypoint `validate_gene(gene, config)` that performs the
core checks described in the README. Returns a list of error dicts. An empty
list means the gene is valid.

Errors are dicts with keys: `path`, `code`, `message`.
"""
from typing import Any, Dict, List

CORE_FACES = ["front", "right", "left", "back"]
BRICK_FACES = ["front", "right", "left"]


def _error(path: str, code: str, message: str) -> Dict[str, str]:
	return {"path": path, "code": code, "message": message}


def validate_gene(gene: Dict[str, Any], config) -> List[Dict[str, str]]:
	"""Validate a gene dict against the core checklist.

	Args:
		gene: parsed JSON gene as dict
		config: module or object providing `MAX_BRICKS` (int)

	Returns:
		list of error dictionaries. Empty list => valid gene.
	"""
	errors: List[Dict[str, str]] = []

	# 1. Top-level structure
	if not isinstance(gene, dict):
		return [ _error("<root>", "invalid_type", "Gene must be a JSON object") ]

	for key in ["id", "brain", "core"]:
		if key not in gene:
			errors.append(_error("<root>", "missing_top_key", f"Missing top-level key: {key}"))

	if errors:
		return errors

	core = gene.get("core")
	if not isinstance(core, dict):
		errors.append(_error("core", "invalid_core", "`core` must be an object"))
		return errors

	# 2. core faces
	for face in CORE_FACES:
		if face not in core:
			errors.append(_error("core", "missing_core_face", f"core missing face: {face}"))
		else:
			if not isinstance(core[face], dict):
				errors.append(_error(f"core.{face}", "invalid_face", "Face must be an object (use {} for empty)"))

	# If core face checks failed, short-circuit
	if errors:
		return errors

	# helper: validate hinge and brick structure recursively
	def validate_brick(node: Dict[str, Any], path: str):
		# node is a brick object
		local_errors: List[Dict[str, str]] = []
		if not isinstance(node, dict):
			local_errors.append(_error(path, "invalid_brick", "Brick must be an object"))
			return local_errors

		# Bricks must only have BRICK_FACES as keys
		for k, v in node.items():
			if k not in BRICK_FACES:
				local_errors.append(_error(f"{path}.{k}", "invalid_brick_face", f"Bricks must only contain faces {BRICK_FACES}, found: {k}"))

		# Ensure each brick face is present (if absent, it's treated as empty)
		# and that non-empty faces contain a hinge with a brick
		for face in BRICK_FACES:
			if face in node and isinstance(node[face], dict) and node[face]:
				hinge = node[face].get("hinge") if isinstance(node[face], dict) else None
				if hinge is None:
					local_errors.append(_error(f"{path}.{face}", "missing_hinge", "Face must contain a hinge object"))
				else:
					# hinge must be dict and contain brick
					if not isinstance(hinge, dict):
						local_errors.append(_error(f"{path}.{face}.hinge", "invalid_hinge", "hinge must be an object"))
					else:
						if "brick" not in hinge:
							local_errors.append(_error(f"{path}.{face}.hinge", "missing_brick", "hinge must contain a brick"))
						else:
							if not isinstance(hinge["brick"], dict):
								local_errors.append(_error(f"{path}.{face}.hinge.brick", "invalid_brick", "brick must be an object"))
							else:
								# recurse
								local_errors.extend(validate_brick(hinge["brick"], f"{path}.{face}.hinge.brick"))
		return local_errors

	# Validate the core structure: faces either {} or hinge
	def validate_core(core_node: Dict[str, Any], path: str):
		local_errors: List[Dict[str, str]] = []
		for face in CORE_FACES:
			face_val = core_node.get(face)
			if not isinstance(face_val, dict):
				local_errors.append(_error(f"{path}.{face}", "invalid_face", "Face must be an object (use {} for empty)"))
				continue
			if face_val:
				hinge = face_val.get("hinge")
				if hinge is None:
					local_errors.append(_error(f"{path}.{face}", "missing_hinge", "Non-empty face must contain a hinge"))
				else:
					if not isinstance(hinge, dict):
						local_errors.append(_error(f"{path}.{face}.hinge", "invalid_hinge", "hinge must be an object"))
					else:
						if "brick" not in hinge:
							local_errors.append(_error(f"{path}.{face}.hinge", "missing_brick", "hinge must contain a brick"))
						else:
							if not isinstance(hinge["brick"], dict):
								local_errors.append(_error(f"{path}.{face}.hinge.brick", "invalid_brick", "brick must be an object"))
							else:
								local_errors.extend(validate_brick(hinge["brick"], f"{path}.{face}.hinge.brick"))
		return local_errors

	errors.extend(validate_core(core, "core"))

	# 5. Brick face validity is implicitly checked above during recursion.

	# 8. Module count
	# Count hinges/bricks pairs (approximate modules)
	def count_modules_in_brick(brick_node: Dict[str, Any]) -> int:
		count = 1  # this brick
		for face in BRICK_FACES:
			face_val = brick_node.get(face)
			if isinstance(face_val, dict) and face_val:
				hinge = face_val.get("hinge")
				if isinstance(hinge, dict) and "brick" in hinge and isinstance(hinge["brick"], dict):
					count += count_modules_in_brick(hinge["brick"])
		return count

	total_modules = 0
	# count core + its bricks
	# core is root brick for counting purpose
	def count_core(core_node: Dict[str, Any]) -> int:
		# count core as 1
		total = 1
		for face in CORE_FACES:
			face_val = core_node.get(face)
			if isinstance(face_val, dict) and face_val:
				hinge = face_val.get("hinge")
				if isinstance(hinge, dict) and "brick" in hinge and isinstance(hinge["brick"], dict):
					total += count_modules_in_brick(hinge["brick"])
		return total

	total_modules = count_core(core)
	max_allowed = getattr(config, "MAX_BRICKS", None)
	if max_allowed is not None and total_modules > max_allowed:
		errors.append(_error("<root>", "too_many_modules", f"Found {total_modules} modules, max allowed is {max_allowed}"))

	# 7. Symmetry: compare right and left subtree structure
	def compare_subtrees(right: Dict[str, Any], left: Dict[str, Any], path: str) -> List[Dict[str, str]]:
		errs: List[Dict[str, str]] = []
		# Both must be dicts
		if not isinstance(right, dict) or not isinstance(left, dict):
			errs.append(_error(path, "asymmetry", "Right and left subtrees must be objects"))
			return errs

		# For each brick face presence (empty vs hinge) must match
		for face in BRICK_FACES:
			r_present = bool(right.get(face))
			l_present = bool(left.get(face))
			if r_present != l_present:
				errs.append(_error(f"{path}.{face}", "asymmetry", f"Face presence mismatch on {face}: right={r_present} left={l_present}"))
			else:
				if r_present and l_present:
					# both have hinges; recurse into their bricks
					r_hinge = right[face].get("hinge")
					l_hinge = left[face].get("hinge")
					if not (isinstance(r_hinge, dict) and isinstance(l_hinge, dict)):
						errs.append(_error(f"{path}.{face}", "asymmetry", "One hinge object is missing or invalid"))
					else:
						r_brick = r_hinge.get("brick")
						l_brick = l_hinge.get("brick")
						if not (isinstance(r_brick, dict) and isinstance(l_brick, dict)):
							errs.append(_error(f"{path}.{face}", "asymmetry", "One brick is missing or invalid"))
						else:
							errs.extend(compare_subtrees(r_brick, l_brick, f"{path}.{face}.hinge.brick"))
		return errs

	# perform symmetry check comparing core.right <-> core.left
	right_node = core.get("right", {})
	left_node = core.get("left", {})
	errors.extend(compare_subtrees(right_node, left_node, "core"))

	return errors

