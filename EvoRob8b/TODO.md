# To do

- Ka 🌱
- Vebjørn 🐻‍❄️
- Sofie 🪱
- Emma ⛰️

## Code

- [x] Gene generator
 - [x] BFS generation
   - for å generere lemmene
 - [x] max parts 
 - [x] symmetry 
 - [x] orientation
   - [x] revisit symmetry
- [x] higher prob of placing bricks on front than on right and left

Gene validator could enforce invariant during ea runs

- [ ] [Gene validator](./gene_validator.py) 
  - [ ] brick has front, right, left faces
    - [ ] especially spine
  - [ ] hinge has brick
  - [ ] spine symmetry
  - [x] module count
  - [ ] count bricks

- [x] brain representation
- [ ] Track rng if time
- [ ] Save and load to file brain -- Emma

- [x] EA
 - [x] Crossover
 - [x] Mutation
 - [x] Inner learning loop for brain optimization @brains
   - [x] Mutation
   - [x] Eval
   - [x] Selection
 - [x] Evaluation
 - [x] Tournament selection
 - [ ] Elitism 🐻‍❄️

- [ ] Ploting -- Ka <!-- started class in EA.py -->
  - [ ] Matplotlib
  - [ ] Fitness over generations
  - [ ] Compare own morph with EA the same amount of generation?

## Writing

- [ ] Pseudo code and illustrations
  - [ ] Crossover illustration and pseudo-code 🌱
  - [ ] Mutation 🐻‍❄️
  - [x] Selection 🪱
- [ ] Ethics statement ⛰️
