import pybullet as p
import pybullet_data
import math
import random
import creature
import time
import genome as genlib
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def make_mountain(num_rocks=100, max_size=0.25, arena_size=10, mountain_height=5):
    def gaussian(x, y, sigma=arena_size/4):
        # Return the height of the mountain at position (x, y) using a Gaussian function
        return mountain_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

    for _ in range(num_rocks):
        x = random.uniform(-1 * arena_size/2, arena_size/2)
        y = random.uniform(-1 * arena_size/2, arena_size/2)
        z = gaussian(x, y)  # Height determined by the Gaussian function

        # Adjust the size of the rocks based on height. Higher rocks (closer to the peak) will be smaller.
        size_factor = 1 - (z / mountain_height)
        size = random.uniform(0.1, max_size) * size_factor

        orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
        rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
        rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1])
        rock_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation)

def make_rocks(num_rocks=100, max_size=0.25, arena_size=10):
    for _ in range(num_rocks):
        x = random.uniform(-1 * arena_size/2, arena_size/2)
        y = random.uniform(-1 * arena_size/2, arena_size/2)
        z = 0.5
        size = random.uniform(0.1,max_size)
        orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
        rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
        rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1])
        rock_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation)

def make_arena(arena_size=10, wall_height=1):
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

p.setGravity(0, 0, -10)

arena_size = 20
make_arena(arena_size=arena_size)

# make_rocks(arena_size=arena_size)

mountain_position = (0, 0, -1)
mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
p.setAdditionalSearchPath('shapes/')
# mountain = p.loadURDF("mountain.urdf", mountain_position, mountain_orientation, useFixedBase=1)
# mountain = p.loadURDF("mountain_with_cubes.urdf", mountain_position, mountain_orientation, useFixedBase=1)

mountain_id = p.loadURDF('gaussian_pyramid.urdf', mountain_position, mountain_orientation, useFixedBase=1)

# Generate creature with elite dna
cr = creature.Creature(gene_count=3)
dna = genlib.Genome.from_csv('elite_v2_48.csv')
cr.update_dna(dna)

# Save it to XML
with open('test.urdf', 'w') as f:
    f.write(cr.to_xml())

# Load it into the sim
creature_id = p.loadURDF('test.urdf', (8, 8, 2))

# Set real time simulation
p.setRealTimeSimulation(1)

# Function to calculate forward direction of creature
def calculate_forward_direction(quaternion):
	# Convert quaternion to rotation matrix
	rotation_matrix = p.getMatrixFromQuaternion(quaternion)
	
	# Forward direction is the third column of the rotation matrix
	forward = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
	
	return forward

# Function to calculate angle to mountain
def calculate_angle_to_mountain(creature_position, creature_orientation, mountain_position):
	# Calculate forward direction of creature
	forward_direction = calculate_forward_direction(creature_orientation)

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

previous_distance = 0
current_distance = 0
previous_position = None
current_position = None
displacement = 0
number_of_contacts = 0
    
while True:
    # Get creature position and orientation
    creature_position, creature_orientation = p.getBasePositionAndOrientation(creature_id)

    # Calculate distance to mountain
    distance_to_mountain = np.linalg.norm(np.array(creature_position) - np.array(mountain_position))
    print(f'Distance: {distance_to_mountain}')

    # Calculate angle to mountain
    angle_to_mountain = calculate_angle_to_mountain(creature_position, creature_orientation, mountain_position)

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
        print('TOUCHED')
        number_of_contacts += 1
        height_on_mountain = creature_position[2]

    # Apply motor control
    motors = cr.get_motors()

    for jid in range(p.getNumJoints(creature_id)):
        p.setJointMotorControl2(creature_id, 
                                jid,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=motors[jid].get_output() * 2,
                                force=100 + ((math.cos(angle_to_mountain) + 1) * 20) - (rate_of_change * 5))

    # Calculate fitness
    fitness = (200 / distance_to_mountain) + (rate_of_change * 100) + (displacement * 2) + (number_of_contacts * 5) + (height_on_mountain ** 2 * 100)
    print(f'Fitness: {fitness}')
            
    # Step the simulation
    p.stepSimulation()

    # Sleep to control simulation speed
    time.sleep(1/240)

    # Position camera to follow creature
    new_position, _ = p.getBasePositionAndOrientation(creature_id)
    p.resetDebugVisualizerCamera(10, 0, 220, new_position)