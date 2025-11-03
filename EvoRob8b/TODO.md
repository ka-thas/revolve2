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

- [ ] Gene validator (not used)
  - [ ] brick has front, right, left faces
    - [ ] especially spine
  - [ ] hinge has brick
  - [ ] spine symmetry
  - [x] module count
  - [ ] count bricks

- [x] brain representation
- [x] Track rng if time
- [x] Save and load to file brain -- Emma

- [x] EA
 - [x] Crossover
 - [x] Mutation
 - [x] Inner learning loop for brain optimization @brains
   - [x] Mutation
   - [x] Eval
   - [x] Selection
 - [x] Evaluation
 - [x] Tournament selection
 - [x] Elitism 🐻‍❄️

- [x] Ploting -- Ka <!-- started class in EA.py -->
  - [x] Matplotlib
  - [x] Fitness over generations
  - [ ] Compare own morph with EA the same amount of generation?
  - [ ] Average over multiple runs
    - SAME PARAMETERS

## Writing

- [ ] Pseudo code and illustrations
  - [x] Crossover illustration and pseudo-code 🌱
  - [x] Mutation 🐻‍❄️
  - [x] Selection 🪱
- [x] Ethics statement ⛰️
- [ ] Explain why we cut off Lamarckian
- [ ] Explain parameters
- [ ] Describe the brain, how parallelization works etc


After the paper is finished:
- [ ] Go through the paper, rewrite so the paper remians one style
- [ ] Go out for ramen?? 🍜


# Experiments

- [ ] parse_gene:
  - [ ] final_best_individual_047246 <- slange, men veldig god fitness
  - [ ] 174804_final_best_individual
- [ ] plot 174804
- [ ] https://robin.wiki.ifi.uio.no/index.php?title=Robin-hpc 
  - Dette var den Kyrre sa var lettest å bruke
  - men tror den kun kan ssh fra uio-eduroam
- [ ] https://robin.wiki.ifi.uio.no/index.php?title=Deep_Learning_Workstations


Hyperparametere lamarkian evoulution
https://www.nature.com/articles/s41598-023-48338-4/tables/1

