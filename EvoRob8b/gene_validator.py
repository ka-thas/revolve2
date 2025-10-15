""" see readme """

def check_brick_faces(gene):
	""" Check that all bricks have front, right, and left faces, especially on the spine. """
	# Implementation goes here
	for brick in gene.get("core", {}).get("bricks", []):
		if "front" not in brick or "right" not in brick or "left" not in brick:
			return False
	return True

def check_hinge_has_brick(gene):
	""" Check that all hinges have an associated brick. """
	# Implementation goes here
	for hinge in gene.get("core", {}).get("hinges", []):
		if "brick" not in hinge:
			return False
	return True

def check_spine_symmetry(gene):
	""" Check that the spine is symmetric. """
	