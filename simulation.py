import pybullet as p
from multiprocessing import Pool
import numpy as np
import math

class Simulation: 
    def __init__(self, sim_id=0):
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id

    def run_creature(self, cr, iterations):
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

        p.setGravity(0, 0, -10, physicsClientId=pid)

        # plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pid)
        # floor = p.createMultiBody(plane_shape, plane_shape, physicsClientId=pid)

        arena_size = 20
        wall_height = 1
        wall_thickness = 0.5
        floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
        floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1])
        floor_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])

        wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
        wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

        # Create four walls
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2])

        wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
        wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2])
        
        mountain_position = (0, 0, -1)
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
        p.setAdditionalSearchPath('shapes/')
        mountain_id = p.loadURDF('gaussian_pyramid.urdf', mountain_position, mountain_orientation, useFixedBase=1)

        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()

        with open(xml_file, 'w') as f:
            f.write(xml_str)
        
        creature_id = p.loadURDF(xml_file, (8, 8, 2), physicsClientId=pid)

        total_distance_to_mountain = 0
        previous_distance = 0
        current_distance = 0
        total_rate_of_change = 0
        previous_position = None
        current_position = None
        displacement = 0
        total_displacement = 0
        number_of_contacts = 0

        for step in range(iterations):
            # Get creature position and orientation
            creature_position, creature_orientation = p.getBasePositionAndOrientation(creature_id, physicsClientId=pid)
            
            # Calculate distance to mountain
            distance_to_mountain = np.linalg.norm(np.array(creature_position) - np.array(mountain_position))

            # Calculate angle to mountain
            angle_to_mountain = self.calculate_angle_to_mountain(creature_position, creature_orientation, mountain_position)

            # Calculate rate of change (positive = correct direction, negative = wrong direction)
            previous_distance = current_distance
            current_distance = distance_to_mountain
            rate_of_change = -((current_distance - previous_distance) / 0.1)

            # Calculate displacement
            previous_position = current_position
            current_position = creature_position

            if previous_position and current_position:
                displacement = abs(np.linalg.norm(np.array(previous_position) - np.array(current_position)))
            
            # Calculate number of contacts and height on mountain
            contact_points = p.getContactPoints(creature_id, mountain_id)
            height_on_mountain = 0

            if contact_points:
                number_of_contacts += 1
                height_on_mountain = creature_position[2]

            # Update motors
            if step % 24 == 0:
                # Apply motor control
                motors = cr.get_motors()

                for jid in range(p.getNumJoints(creature_id,
                                                physicsClientId=self.physicsClientId)):
                    p.setJointMotorControl2(creature_id, 
                                            jid,
                                            controlMode=p.VELOCITY_CONTROL,
                                            targetVelocity=motors[jid].get_output() * 2,
                                            force=100 + ((math.cos(angle_to_mountain) + 1) * 20) - (rate_of_change * 5),
                                            physicsClientId=self.physicsClientId)
            
            # Step the simulation
            p.stepSimulation(physicsClientId=pid)
            
        # Update fitness score
        cr.update_fitness_score(200 / (total_distance_to_mountain / iterations)
                                + (total_rate_of_change / iterations) * 100
                                + (total_displacement / iterations) * 2
                                + number_of_contacts * 5
                                + height_on_mountain ** 2 * 100)

    # Function to calculate forward direction of creature
    def calculate_forward_direction(self, quaternion):
        # Convert quaternion to rotation matrix
        rotation_matrix = p.getMatrixFromQuaternion(quaternion)
        
        # Forward direction is the third column of the rotation matrix
        forward = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
        
        return forward

    # Function to calculate angle to mountain
    def calculate_angle_to_mountain(self, creature_position, creature_orientation, mountain_position):
        # Calculate forward direction of creature
        forward_direction = self.calculate_forward_direction(creature_orientation)

        # Calculate direction to mountain
        direction_to_mountain = [mountain_position[i] - creature_position[i] for i in range(3)]

        # Normalise the direction vectors
        forward_direction_norm = np.linalg.norm(forward_direction)
        direction_to_mountain_norm = np.linalg.norm(direction_to_mountain)
        forward_direction = [f / forward_direction_norm for f in forward_direction]
        direction_to_mountain = [d / direction_to_mountain_norm for d in direction_to_mountain]

        # Calculate dot product and determinant
        dot_product = np.dot(forward_direction[:2], direction_to_mountain[:2])
        determinant = forward_direction[0] * direction_to_mountain[1] - forward_direction[1] * direction_to_mountain[0]

        # Calculate angle using arctan2
        angle = np.arctan2(determinant, dot_product)

        # Ensure angle is in range (0, 2pi)
        if angle < 0:
            angle += 2 * math.pi

        return angle
        
    # You can add this to the Simulation class:
    def eval_population(self, pop, iterations):
        for cr in pop.creatures:
            self.run_creature(cr, 2400) 

class ThreadedSim():
    def __init__(self, pool_size):
        self.sims = [Simulation(i) for i in range(pool_size)]

    @staticmethod
    def static_run_creature(sim, cr, iterations):
        sim.run_creature(cr, iterations)
        return cr
    
    def eval_population(self, pop, iterations):
        """
        pop is a Population object
        iterations is frames in pybullet to run for at 240fps
        """
        pool_args = [] 
        start_ind = 0
        pool_size = len(self.sims)
        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):# the end
                    break
                # work out the sim ind
                sim_ind = i % len(self.sims)
                this_pool_args.append([
                            self.sims[sim_ind], 
                            pop.creatures[i], 
                            iterations]   
                )
            pool_args.append(this_pool_args)
            start_ind = start_ind + pool_size

        new_creatures = []
        for pool_argset in pool_args:
            with Pool(pool_size) as p:
                # it works on a copy of the creatures, so receive them
                creatures = p.starmap(ThreadedSim.static_run_creature, pool_argset)
                # and now put those creatures back into the main 
                # self.creatures array
                new_creatures.extend(creatures)
        pop.creatures = new_creatures
