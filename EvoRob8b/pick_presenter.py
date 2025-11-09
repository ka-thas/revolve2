import random

subjects= ["intro", "methods", "setup", "results", "discussion", "conclusion"]
presenters = ["Emma", "Ka", "Sofie", "Vebjørn"]
random.shuffle(presenters)

assignments = [(subject, presenters[i % len(presenters)]) for i, subject in enumerate(subjects)]

for subject, presenter in assignments:
    print(f"{subject}: {presenter}")