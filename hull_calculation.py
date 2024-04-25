from scipy.spatial import ConvexHull
import numpy as np
from collections import Counter
import logging

eps = 1e-5

def is_facing_exterior(equation, points):
    for i, point in enumerate(points):
        value = np.sum(equation[:-1]*point) + equation[-1]
        if value > 1e-4:
            return False
        elif value < 1e-4:
            return True
    raise ValueError("All points are on a single face")

def energy_distance_to_hyperplane(equation, point):
    denominator = equation[-2]
    denominator += eps if denominator > 0 else -eps
    return (np.sum(equation[:-1]*point)+equation[-1])/denominator

def distance_to_hull_equations(equations, point):
    distances = [energy_distance_to_hyperplane(equation, point) for equation in equations]
    min_i = np.argmin(distances)
    return distances[min_i], min_i

def filter_equations(equations, points):
    # the equations' normals have to face the exterior of the hull
    indices = []
    for i, equation in enumerate(equations):
        to_exterior_sign = 1 if is_facing_exterior(equation, points) else -1
        
        # filter almost vertical or upper faces of the hull
        if to_exterior_sign*equation[-2] < -eps:
            indices.append(i)
    return equations[indices], indices

def distance_to_hull_matrix(data, point):
    hull = ConvexHull(data)
    equations, filter_index = filter_equations(hull.equations, data)
    distance, min_i = distance_to_hull_equations(equations, point)
    min_i = filter_index[min_i]
    simplex = hull.simplices[min_i]
    return distance, simplex

def get_proportions(composition):
    result = dict(Counter(composition))
    total = len(composition)
    for element in result:
        result[element] /= total
    return result

def get_data_matrix(data, element_list):
    data_matrix = np.empty((len(data), len(element_list)), dtype=float)
    for struct_i, structure in enumerate(data):
        proportions = get_proportions(structure['A'])
        for element_i, element in enumerate(element_list[:-1]):
            data_matrix[struct_i, element_i] = proportions[element] if element in proportions else 0
            data_matrix[struct_i, -1] = structure['E']
    return data_matrix

def get_decompose_proportions(data_matrix, simplex_index, target):
    M = data_matrix[simplex_index]
    M[:,-1] = 1 - np.sum(M[:,:-1], axis=1)
    M = M.T
    y = target
    y[-1] = 1 - np.sum(y[:-1])
    x = np.linalg.lstsq(M, y, rcond=None)[0].tolist()
    return x

def distance_to_hull(data, composition, energy):
    # Return the distance to hull of data[indices] from the point with given composition and energy 
    total = np.sum(list(composition.values()))
    for element in composition:
        composition[element] /= total
        
    logging.info(f"Composition requested: {composition}")
    
    elements = list(composition.keys())
    data_matrix = get_data_matrix(data, elements)
    
    point = np.empty(len(composition))
    point[:-1] = [composition[element] for element in elements[:-1]]
    point[-1] = energy
    
    distance, simplex = distance_to_hull_matrix(data_matrix, point)
    decompose_to = get_decompose_proportions(data_matrix, simplex, point)
    
    return distance, simplex, decompose_to