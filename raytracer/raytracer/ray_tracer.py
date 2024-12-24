import argparse
from PIL import Image
import numpy as np
import os

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

# Try to import joblib for parallel work
parallel = True
try:
    from joblib import Parallel, delayed
except ImportError as e:
    parallel = False

"""
    A class to hold a light ray's hit on a surface information.
"""
class Hit:
    def __init__(self, distance, surface, point):
        self.distance = distance  # Distance from the ray origin to the intersection point
        self.surface = surface  # The surface that the ray intersects
        self.point = point  # The intersection point in 3D space
        self.normal = normalize(self.point - surface.position) if type(surface) != InfinitePlane else normalize(surface.normal)

""" Parse the objects into 3 different type of objects (lights, materials and surfaces)"""
def parse_objects(objects):
    # Declare variables
    lights = []
    materials = []
    surfaces = []

    # Iterate through all the objects and classify them
    for object in objects:
        if type(object) is Light:
            lights.append(object)
        elif type(object) is Material:
            materials.append(object)
        else:
            surfaces.append(object)

    return lights, materials, surfaces

"""   Normalize a list of vectors and return them"""
def normalize(vecList):

    if vecList.ndim == 1:
        norm = np.linalg.norm(vecList)
        if norm == 0:
            return vecList
        return vecList / norm
    else:
         norm = np.linalg.norm(vecList, axis=2)
    return vecList / norm[..., np.newaxis]

"""
    Calculate the normalized parameters of the camera and return the center point , right and high vectors an the asspect ration"""

def normalize_camera(camera: Camera, width):
    # Normalize the look at vector

    normmalizedLookAtVector = normalize(camera.look_at - camera.position)
    # Calculate the center point of the camera
    centerPoint = camera.position + camera.screen_distance * normmalizedLookAtVector

    # Normalize up and right vectors
    rightVector = normalize(np.cross(camera.up_vector, normmalizedLookAtVector))
    highVector = normalize(np.cross(normmalizedLookAtVector, rightVector))

    # Calculate the aspect ratio of the camera
    ratio = camera.screen_width / width

    return (centerPoint, rightVector, highVector, ratio)



""" Calculate the intersections with a cube and return the distances of the intersection points as an array """
def calc_cube_intersections(rays: np.ndarray, cameraPos, cube: Cube):
    # Initialize arrays to store the min and max intersection distances for each axis
    minDist = np.full_like(rays, -np.inf)
    maxDist = np.full_like(rays, np.inf)

    # Iterate over each axis (x, y, z)
    for i in range(3):
        # Calculate the positive and negative faces of the cube
        negativeFace = cube.position[i] - (cube.scale / 2)
        positiveFace = cube.position[i] + (cube.scale / 2)


        # Mask to check for non-zero ray direction components
        mask = np.abs(rays[..., i]) >= 0

        # Calculate intersection distances with the negative and positive faces of the cube
        distnace1 = np.where(mask, (negativeFace - cameraPos[..., i]) / rays[..., i], np.inf)
        distnace2 = np.where(mask, (positiveFace - cameraPos[..., i]) / rays[..., i], -np.inf)

        # Update t_min and t_max for the current axis (between the positive and negative faces distance)
        minDist[..., i] = np.minimum(distnace1, distnace2)
        maxDist[..., i] = np.maximum(distnace1, distnace2)

    # Calculate the entry and exit distances for the ray intersection with the cube
    distanceStart = np.amax(minDist, axis=-1)
    distanceEnd = np.amin(maxDist, axis=-1)

    # Choose the valid intersection distance
    distance = np.where(distanceStart <= distanceEnd, distanceStart, np.inf)

    return distance
"""Calculate intersections with a sphere and return the distance of the intersections points from the camera"""
def calc_sphere_intersections(rays: np.ndarray, cameraPos, sphere: Sphere):
    # Calculate the vector from the origin of the ray to the center of the sphere
    z = cameraPos - sphere.position

    # Calculate the coefficient of the qudratic equation
    x = 2 * np.sum(rays * z, axis=-1)

    # Calculate the constant term of the quadratic equation
    y = np.sum(z**2, axis=-1) - sphere.radius**2

    # Calculate the discriminant of the quadratic equation
    d = x**2 - 4 * y

    # Calculate the two possible intersection distances
    sqrtD = np.sqrt(np.maximum(d, 0))
    distance1 = (-x + sqrtD) / 2
    distance2 = (-x - sqrtD) / 2

    # Choose the closer valid intersections distance (from the 2 possible ones)
    distance = np.where(d >= 0, np.minimum(distance1, distance2), np.inf)

    return distance

"""Calculate intersections with an infinite plane and return the distances of the intersection point from the camera as an array"""
def calc_infinite_plane_intersections(rays: np.ndarray, cameraPos, plane: InfinitePlane):
    # Calculate the dot product of the ray direction and the plane normal
    dotProduct = np.sum(plane.normal * rays, axis=-1)

    # Calculate all the valid hits (exclued rays parallel to the plane)
    validHits = np.abs(dotProduct) >= 1e-6

    # Calculate intersection distance
    distance = np.where(validHits, np.dot(plane.offset - cameraPos, plane.normal) / dotProduct, np.inf)

    return distance
""" Calculate the rays intersection with the screen and return the result as an array"""
def calc_ray_intersction_with_screen(camera: Camera, center, highVector: np.ndarray, rightvector: np.ndarray, ratio, iVector, jVector, width, height):
    # Reshape the right vector and the up vector to be (1, 1, 3)

    highVector = highVector.reshape(1, 1, 3)
    rightvector = rightvector.reshape(1, 1, 3)
    # Calculate the intersections with the screen
    screen = (center + np.expand_dims((jVector - width//2), 2) *ratio * rightvector - np.expand_dims((iVector - height//2), 2) * ratio * highVector)

    # Normalize the vectors and return them
    return normalize(screen-camera.position)


"""Calculate the camera rays intersection with the surfaces in the scene and return them as an array"""
def calc_intersections(rays: np.ndarray, surfaces, cameraPos):
    # Set numpy to ignore divide and invalid errors
    np.seterr(divide='ignore', invalid='ignore')
    hitsList = []
    raysHits = np.empty(rays.shape[:2], dtype=np.ndarray)
    lightHits = []

    #checking if the curface is an instance of cube , infinite plane or a sphere and append it to the hits list
    for surface in surfaces:
        if isinstance(surface, Cube):
            hitsList.append((calc_cube_intersections(rays, cameraPos, surface), surface))
        else:
            if isinstance(surface, InfinitePlane):
               hitsList.append((calc_infinite_plane_intersections(rays, cameraPos, surface), surface))
            else:
                  if isinstance(surface, Sphere):
                      hitsList.append((calc_sphere_intersections(rays, cameraPos, surface), surface))



    # Aggregate all the rays hits to a single list
    if hitsList and rays.ndim > 1: # If there are many rays
        for i in range(rays.shape[0]):
            for j in range(rays.shape[1]):
                curRay = rays[i][j]
                curRayHits = []
                # Iterate through all the hits
                for distance, surface in hitsList:
                    distance_ij = distance[i][j]
                    # Make sure distance is not zero (with epsilon) and not infinity
                    if 0.000001 < distance_ij < np.inf:
                        hitPoint = cameraPos + distance_ij * curRay
                        curRayHits.append(Hit(distance_ij, surface, hitPoint))

                # If there were hits with the current ray
                if curRayHits:
                    raysHits[i][j] = sorted(curRayHits, key=lambda x: x.distance)
        # Return the calculated ray hits with the scene's surfaces
        return raysHits

    elif hitsList: # If there is only one ray
        # Iterate through all the hits
        for distance, surface in hitsList:
            # Make sure distance is not zero (with epsilon) and not infinity
            if 0.000001 < distance < np.inf:
                hitPoint = cameraPos + distance * rays
                lightHits.append(Hit(distance, surface, hitPoint))

        # Sort the light hits list and return it
        return sorted(lightHits, key=lambda x: x.distance)

        # Return None if there were no intersection between the rays and the surfaces
    return None

"""Calculate the light intensity for a specific light and return it"""
def calc_light_intesity(hit: Hit, light: Light, surfaces, settings: SceneSettings):
    shadowRaysNumber = int(settings.root_number_shadow_rays)

    # Calculate direction vector from the light to the hit point
    lightAngle = normalize(hit.point - light.position)

    # Determine a suitable dimension for constructing orthogonal vectors
    primaryAxis = 1 if np.argmax(np.abs(lightAngle)) != 0 else 0

    # Construct right and up vectors for the light's local coordinate system
    rightVector = normalize(np.cross(lightAngle, np.array([primaryAxis, 1 - primaryAxis, 0]))).reshape(1, 1, 3)
    highVector = normalize(np.cross(lightAngle, rightVector)).reshape(1, 1, 3)

    # Initialize the count of successful light ray hits
    hitsCounter = 0

    # Calculate the size of each cell in the shadow rays grid
    cellSize = light.radius / shadowRaysNumber
    gridIndices = np.arange(shadowRaysNumber) - shadowRaysNumber // 2

    # Create a grid of points around the light source
    jInd, iInd = np.meshgrid(gridIndices, gridIndices)
    lightGridPoints = (light.position +np.expand_dims(jInd, 2) * cellSize * rightVector - np.expand_dims(iInd, 2) * cellSize * highVector)

    # Add random jitter within each cell to simulate area light
    rightJitter = np.expand_dims(np.random.uniform(-0.5 * cellSize, 0.5 * cellSize, size=(shadowRaysNumber,  shadowRaysNumber)), 2) * rightVector
    highJitter = np.expand_dims(np.random.uniform(-0.5 * cellSize, 0.5 * cellSize, size=(shadowRaysNumber, shadowRaysNumber)), 2) * highVector
    lightGridPoints += rightJitter + highJitter

    # Calculate direction vectors from each grid point to the hit point
    rayAngles = normalize(hit.point - lightGridPoints)

    # Find intersections of the rays with the surfaces
    light_ray_intersections = calc_intersections(rayAngles, surfaces, lightGridPoints)

    # Count the number of rays that directly hit the target surface
    matchingSurfaces = np.array([hit_obj[0].surface if hit_obj is not None else None for hit_obj in light_ray_intersections.ravel()]) == hit.surface
    hitsCounter = np.count_nonzero(matchingSurfaces)

    # Calculate and return the light intensity at the hit point
    intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (hitsCounter / (shadowRaysNumber**2))
    return intensity


"""function that calls calculate_color"""
def calc_color(hits, cameraPos, materials, lights, surfaces, settings: SceneSettings, recursionDepth):
    color = calculate_color(hits, cameraPos, materials, lights, surfaces, settings, recursionDepth)
    return color

"""Calculate the color at the ray hit point based on the ligths,recursive reflections and matrials and return it"""
def calculate_color(hits, cameraPos, materials, lights, surfaces, settings: SceneSettings, recursionDepth):
    # Return background color if no hits or maximum recursion depth is exceeded
    if not hits or recursionDepth > settings.max_recursions:
        return settings.background_color

    # Initialize colors for diffuse and specular components
    diffuseColor = np.zeros(3)
    specularColor = np.zeros(3)

    # Get the nearest hit
    hit = hits[0]
    material = materials[hit.surface.material_index - 1]

    for light in lights:
        # Calculate direction vector from the hit point to the light
        angle = normalize(light.position - hit.point)

        # Calculate light intensity considering shadows
        intensity = calc_light_intesity(hit, light, surfaces, settings)

        # Calculate diffuse color contribution using Lambertian reflection model
        diffuseContrib = np.maximum(np.dot(hit.normal, angle), 0)
        diffuseColor += material.diffuse_color * diffuseContrib * light.color * intensity

        # Calculate specular color contribution using Phong reflection model
        angle2 = normalize(cameraPos - hit.point)
        halfDir = normalize(angle + angle2)
        specularContrib = np.power(np.maximum(np.dot(hit.normal, halfDir), 0), material.shininess)
        specularColor += specularContrib * material.specular_color * light.specular_intensity * light.color * intensity

    # Handle transparency by recursively reducing the list of hits
    bgColor = 0 if material.transparency <= 0 else calculate_color(hits[1:], cameraPos, materials, lights, surfaces, settings, recursionDepth)

    # Handle reflection by calculating the reflection color
    reflectionColor = calc_reflection_color(hit, cameraPos, materials, lights, surfaces, settings, material, recursionDepth)

    # Sum all color contributions and apply transparency
    endColor = (bgColor * material.transparency +
                   (diffuseColor + specularColor) * (1 - material.transparency) +
                   reflectionColor)

    # Ensure the color values are within the valid range [0, 1]
    return np.clip(endColor, 0, 1)

""" Calculate reflection color and return it"""
def calc_reflection_color(hit: Hit, cameraPos, materials, lights, surfaces, settings: SceneSettings, material: Material, recursionDepth):
    # Normalize the direction from the hit point to the ray origin (camera)
    camAngle = normalize(cameraPos - hit.point)

    # Calculate the reflected ray direction
    reflectedDirection = 2 * np.dot(hit.normal, camAngle) * hit.normal - camAngle

    # Find intersections of the reflected ray with the surfaces
    reflectionHits = calc_intersections(reflectedDirection, surfaces, hit.point)

    # If no intersections are found, return the background color modulated by the reflection color of the material
    if not reflectionHits:
        return settings.background_color * material.reflection_color

    # Calculate the color at the intersection point of the reflected ray
    reflectionColor = calculate_color(reflectionHits, hit.point, materials, lights, surfaces, settings, recursionDepth+ 1)

    # Return the reflection color modulated by the material's reflection color
    return reflectionColor * material.reflection_color

"""Render the scene by tracing rays from the camera through each pixel and calculating the color based on intersections
    with objects and return the redered image as numpy array od shapes"""
def rendering(camera: Camera, scene_settings: SceneSettings, height, width, objects):
    # Initialize the image array with zeros
    img = np.zeros((height, width, 3))

    # Split objects into materials, surfaces, and lights
    lights, materials, surfaces = parse_objects(objects)

    # Get normalized camera parameters
    center, rightVector, highVector, aspectRatio = normalize_camera(camera, width)

    # Create meshgrid for pixel coordinates
    iVector, jVector = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Generate rays for each pixel
    rays = calc_ray_intersction_with_screen(camera, center, highVector, rightVector, aspectRatio, iVector, jVector, width, height)

    # Calculate intersections of rays with surfaces
    hits = calc_intersections(rays, surfaces, camera.position)

    # If joblib is installed on the machine
    if (parallel):
        # Use parallel processing to calculate the color for each pixel
        results = Parallel(n_jobs=os.cpu_count())(delayed(calc_color)(hits[i][j], camera.position, materials, lights, surfaces, scene_settings, 1)
            for i in range(height) for j in range(width))

        # Assign the calculated colors to the image array
        for index, color in enumerate(results):
            i = index // width
            j = index % width
            img[i, j] = color * 255

    # If joblib is not installed on the machine
    else:
        for i in range(height):
            for j in range(width):
                img[i][j] = calculate_color(hits[i, j], camera.position, materials, lights, surfaces, scene_settings, 1) * 255

    return img

"""Save the rendered image as a png file"""
def saveRenderedImage(image_array, path):

    image = Image.fromarray(np.uint8(image_array))
    image.save(path)


"""Parse the scene file"""
def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(np.asarray(params[:3]), np.asarray(params[3:6]), np.asarray(params[6:9]), params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(np.asarray(params[:3]), params[3], params[4])
            elif obj_type == "mtl":
                material = Material(np.asarray(params[:3]), np.asarray(params[3:6]), np.asarray(params[6:9]), params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(np.asarray(params[:3]), params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(np.asarray(params[:3]), params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(np.asarray(params[:3]), params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(np.asarray(params[:3]), np.asarray(params[3:6]), params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Implementation the ray tracer
    imageArr = rendering(camera, scene_settings, args.height, args.width, objects)

    # Save the output image
    saveRenderedImage(imageArr, args.output_image)

if __name__ == '__main__':
    main()