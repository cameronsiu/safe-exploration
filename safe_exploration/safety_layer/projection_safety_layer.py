import numpy as np

def project_vector_onto_vector(v, u):
    """
    Projects vector v onto vector u.

    Args:
        v (np.ndarray): The vector to be projected.
        u (np.ndarray): The vector onto which v is projected.

    Returns:
        np.ndarray: The projection of v onto u.
    """
    dot_product_vu = np.dot(v, u)
    dot_product_uu = np.dot(u, u)
    projection = (dot_product_vu / dot_product_uu) * u
    return projection

def project_vector_onto_plane(v, normal_vector, point_on_plane):
    """
    Projects vector v onto a plane defined by a normal vector and a point.

    Args:
        v (np.ndarray): The vector to be projected.
        normal_vector (np.ndarray): The normal vector of the plane.
        point_on_plane (np.ndarray): A point lying on the plane.

    Returns:
        np.ndarray: The projection of v onto the plane.
    """
    # Vector from the point on the plane to the vector's origin (if v is a position vector)
    # Or, if v is a direction vector, this step is slightly different.
    # For a position vector v, the vector from the plane to v is (v - point_on_plane).
    # We project this difference vector onto the normal.
    vector_to_project = v - point_on_plane
    
    # Project the vector_to_project onto the normal vector
    projection_onto_normal = project_vector_onto_vector(vector_to_project, normal_vector)
    
    # Subtract this projection from the original vector (relative to the plane's point)
    # and add back the point on the plane to get the absolute projected position.
    projection_onto_plane = v - projection_onto_normal
    return projection_onto_plane



class ProjectionSafetyLayer:

    def __init__(self, config):
        self._margin = config.target_margin
        self._lidar_direction_cache = {}

    def _get_lidar_directions(self, number_of_rays):
        if number_of_rays in self._lidar_direction_cache:
            return self._lidar_direction_cache[number_of_rays]

        lidar_directions = np.zeros((number_of_rays, 2))
        spacing = 2 * np.pi / number_of_rays
        for i in range(number_of_rays):
            # add 0.001 so there aren't flat/vertical lines
            angle = i * spacing + 0.001
            x = np.cos(angle)
            y = np.sin(angle)
            lidar_directions[i] = np.array([x, y])

        self._lidar_direction_cache[number_of_rays] = lidar_directions
        return lidar_directions

    def get_safe_action(self, observation, action, c):
        original_action = action
        # Subtract a margin for the moving obstacles
        lidar_readings = observation["lidar_readings"] - 0.02
        lidar_directions = self._get_lidar_directions(lidar_readings.shape[0])

        for i in range(lidar_readings.shape[0]):
            direction = lidar_directions[i]
            distance = lidar_readings[i]

            # Only need to do this if they are facing the same way
            if np.dot(direction, action) > 0:
                action_along_lidar = project_vector_onto_vector(action, direction)
                length_along_lidar = np.linalg.norm(action_along_lidar)
                # Only need to do this if we risk moving into the object
                if length_along_lidar > distance:
                    to_subtract = action_along_lidar * (length_along_lidar - distance) / length_along_lidar
                    action = action - to_subtract
            
        return action
