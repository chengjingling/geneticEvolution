# If you on a Windows machine with any Python version 
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# the multi-threaded version does not work
# so instead, you can use this version. 

import unittest
import population
import simulation 
import genome 
import creature 
import numpy as np
import random

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=30, 
                                    gene_count=3)
        sim = simulation.Simulation()

        for iteration in range(50):
            # Run each creature in simulation
            for cr in pop.creatures:
                sim.run_creature(cr, 1200)          

            # Get fitness of each creature
            fits = [cr.fitness_score 
                    for cr in pop.creatures]

            # Get number of links for each creature
            links = [len(cr.get_expanded_links()) 
                     for cr in pop.creatures]

            print(iteration, "fittest:", np.round(np.max(fits), 3), 
                  "mean:", np.round(np.mean(fits), 3), "mean links", np.round(np.mean(links)), "max links", np.round(np.max(links)))       
            
            # Get fitness map
            fit_map = population.Population.get_fitness_map(fits)

            # Get fittest creature
            fittest_creature_index = np.argmax(fits)
            fittest_creature = pop.creatures[fittest_creature_index]

            # Keep fittest creature
            new_creatures = [fittest_creature]

            for i in range(len(pop.creatures)):
                if (pop.creatures[i] == fittest_creature): ConnectionRefusedError
                
                # Select parents (one of them is always the fittest creature)
                p1 = fittest_creature
                p2_ind = population.Population.select_parent(fit_map)
                p2 = pop.creatures[p2_ind]

                # Crossover
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)

                # Create new creature with mutated DNA
                cr = creature.Creature(gene_count=1)
                cr.update_dna(dna)
                new_creatures.append(cr)

            # Replace 5 of the new creatures with random creatures
            random_count = 5   
            random_indexes = random.sample(range(1, len(pop.creatures)), random_count)

            for i in range(random_count):
                new_creatures[random_indexes[i]] = creature.Creature(gene_count=3)

            # Elitism
            max_fit = np.max(fits)
            
            for cr in pop.creatures:
                if cr.fitness_score == max_fit:
                    filename = "elite_v2_" + str(iteration) + ".csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break
            
            pop.creatures = new_creatures
                            
        self.assertNotEqual(fits[0], 0)

unittest.main()