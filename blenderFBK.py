import bpy
from pathlib import Path
import json
import numpy as np
from math import sqrt, radians, tan, atan
import mathutils
from mathutils import Vector


# Class with properties, that are identical for all light types
class Light:
    def __init__(self, name: str, location: tuple, energy: float, color: tuple, rotation_euler: tuple) -> None:
        self.name = name
        self.location = location
        self.energy = energy
        self.color = color
        self.rotation_euler = rotation_euler


# Class for pointlights, class specific proeprties can be added if necessary
class AreaLight(Light):
    type = "AREA"

    def __init__(self, name, location, energy, color, size, rotation_euler) -> None:
        self.size = size
        super().__init__(name, location, energy, color, rotation_euler)

    @classmethod
    def from_dict(cls, data_dict: dict):
        return cls(name=data_dict["name"], location=data_dict["location"], energy=data_dict["energy"], color=data_dict["color"], size=data_dict["size"], rotation_euler=data_dict["rotation_euler"])


class LightSetup():
    def __init__(self) -> None:
        print(f"Initializing lightsetup...")
        self.data = {}
        self.reload()
        if len(self.data) > 0:
            print(f"Initialized lightsetup.")
        else:
            print(f"Initialized empty lightsetup.\n")

    def reload(self):
        if len(get_objects_of_type("LIGHT")) > 0: 
            self.data = self._collect_lights()
        else:
            self.data = {}

    # returns all the lights in the current scene as a lightsetup dictionary
    def _collect_lights(self) -> dict:
        lights = get_objects_of_type("LIGHT")
        lamps = {}
        if len(lights) > 0:     
            lamps["type"] = "lightsetup"
            lamps["lights"] = {}
            
            print(f"    Found {len(lights)} lights in scene.")

            for light in lights:
                if light.data.type.lower() == "area":
                    arealight = AreaLight(name=light.name, location=tuple(light.location), energy=light.data.energy, color= tuple(light.data.color), size=light.data.size, rotation_euler=tuple(light.rotation_euler))
                    lamps["lights"][arealight.name] = arealight.__dict__
                    lamps["lights"][arealight.name].update({"type": arealight.type})
                    
                    print(f"    Added light: {light.data.name} of type {light.data.type.lower()} to lightsetup")
                else:
                    print(f"    Light of type {light.data.type.lower()} was ignored.")
        else:
            print("Couldn't find any lights in the scene.\n")

        return lamps

    # Adds lights from the lightsetup data to the scene
    def create(self):
        if len(self.data) > 0:
            lights = self.data["lights"]
            for key in lights:
                if lights[key]["type"].lower() == "area":
                    lamp = AreaLight.from_dict(lights[key])
                    self._add_area_lamp(lamp)
                    print(f"    Added lamp: \"{lamp.name}\" of type {lamp.type.lower()} to scene.")
                else:
                    print("Only Lights from type area are supported at the moment.")
        else:
            print("Can't create an empty lightsetup. Add lights zu lightsetup.")

    def clear_scene(self):
        select_objects_of_type(type="LIGHT")
        delete_selected_objects()
        print(f"Deleted all light sources from the scene.")
            
            
    # Creates lights in blender scene from dictionary cotaining the data
    def from_dict(self, data: dict):
        if data["type"] == "lightsetup":
            self.data=data
            #self.data=data["lights"]
            print(f"Added lightsetup consisting of {len(self.data)} lamp(s).")
        
    def from_json(self, filepath):
        with open(filepath, "r") as json_file:
            config = json.load(json_file)
        print(f"Loaded {Path(filepath).name}.")

        if config["type"] == "lightsetup":
            self.data=config
            print(f"Found lightsetup in {Path(filepath).name}.")
            
    def export_json(self, directory:str, filename:str):
        filepath = Path(directory).joinpath(f"{Path(filename).stem}.json")
        if len(self.data) > 0:
            with open(filepath, "w") as outfile:
                json.dump(self.data, outfile, indent = 4)
                print(f"Saved config at :{filepath}.")
        else:
            print("Export aborted.")

    def _add_area_lamp(self,lamp: AreaLight):
        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name=lamp.name, type=lamp.type)

        # create new object with light datablock
        light_object = bpy.data.objects.new(name=lamp.name, object_data=light_data)

        # link light object
        bpy.context.collection.objects.link(light_object)

        # change properties
        light_object.location = lamp.location
        light_object.rotation_euler = lamp.rotation_euler
        light_object.data.energy = lamp.energy
        light_object.data.color = lamp.color
        light_object.data.size = lamp.size


class CameraPath():
    def __init__(self, coordinates:list, method:str, calculation_args:dict = {}) -> None:
        self.type = "camerapath"
        self.method = method
        self.calculation_args = calculation_args
        self.coordinates = coordinates
        
    def as_config_dict(self)-> dict:
        config = {}
        config["type"] = self.type
        config["method"] = self.method
        config["calculation_args"] = self.calculation_args
        config["coordinates"] = self.coordinates

        return config

    @classmethod
    def from_dict(cls, data:dict):
        return cls(coordinates=data["coordinates"], method=data["method"], calculation_args=data["calculation_args"])

    def export_json(self, directory: str, filename: str):
        data = self.__dict__

        # Cast vector coordinates to a json serializable type
        data["coordinates"] = np.asarray(data["coordinates"]).tolist()

        filepath = Path(directory).joinpath(f"{Path(filename).stem}.json")
        if len(data) > 0:
            with open(filepath, "w") as outfile:
              json.dump(data, outfile, indent = 4)
              print(f"Saved camerapath config at :{filepath}.")
        else:
            print("Export aborted.")

    # calculates coordinates based on an icosphere  
    # vertex_positions of type: full_sphere, upper_half, lower_half
    # rotation angle in degree
    def icosphere_coordinates(scale:float= 1.0, subdivisions:int= 2, rotation_angle:float=0.0, vertex_positions:str= "full_sphere"):
            # based on https://sinestesia.co/blog/tutorials/python-icospheres/
        
        def vertex(x, y, z): 
            """ Return vertex coordinates fixed to the unit sphere """ 
            length = sqrt(x**2 + y**2 + z**2) 
            return [(i * scale) / length for i in (x,y,z)]

        def middle_point(point_1, point_2): 
            """ Find a middle point and project to the unit sphere """ 
            # We check if we have already cut this edge first 
            # to avoid duplicated verts 
            smaller_index = min(point_1, point_2) 
            greater_index = max(point_1, point_2) 

            key = '{0}-{1}'.format(smaller_index, greater_index) 

            if key in middle_point_cache: 
                return middle_point_cache[key] 

            # If it's not in cache, then we can cut it 
            vert_1 = verts[point_1] 
            vert_2 = verts[point_2] 
            middle = [sum(i)/2 for i in zip(vert_1, vert_2)] 

            verts.append(vertex(*middle)) 

            index = len(verts) - 1 
            middle_point_cache[key] = index 

            return index
        
        def delete_vertices(area:str, verts):
            tmp_verts = verts
            if area == "upper_half":
                verts = [x for x in tmp_verts if not x[2] <= 0]
            elif area == "lower_half":
                verts = [x for x in tmp_verts if not x[2] >= 0]
            return verts

        # Rotate vertices around z axis
        def rotate(angle, verts)-> list:
            rotated_verts = []
            # Rotation matrix
            mat_rot = mathutils.Matrix.Rotation(radians(angle), 3, 'Z')
            for vert in verts:
                vec = mathutils.Vector((vert[0], vert[1], vert[2]))
                rotated_verts.append(mat_rot @ vec)

            return rotated_verts
        
        middle_point_cache = {}
        
        # Make the base icosahedron
        PHI = (1 + sqrt(5)) / 2
        
        verts = [ vertex(-1, PHI, 0), 
                vertex( 1, PHI, 0), 
                vertex(-1, -PHI, 0), 
                vertex( 1, -PHI, 0), 
                
                vertex(0, -1, PHI), 
                vertex(0, 1, PHI), 
                vertex(0, -1, -PHI), 
                vertex(0, 1, -PHI), 
                
                vertex( PHI, 0, -1), 
                vertex( PHI, 0, 1), 
                vertex(-PHI, 0, -1), 
                vertex(-PHI, 0, 1), ]
        
        faces = [ 
            # 5 faces around point 0 
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], 
            
            # Adjacent faces 
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], 
            
            # 5 faces around 3 
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], 
            
            # Adjacent faces 
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1], 
        ]
        
        # Subdivisions 
        for i in range(subdivisions): 
            faces_subdiv = [] 
            
            for tri in faces: 
                v1 = middle_point(tri[0], tri[1]) 
                v2 = middle_point(tri[1], tri[2]) 
                v3 = middle_point(tri[2], tri[0])

                faces_subdiv.append([tri[0], v1, v3]) 
                faces_subdiv.append([tri[1], v2, v1]) 
                faces_subdiv.append([tri[2], v3, v2]) 
                faces_subdiv.append([v1, v2, v3]) 
                
            faces = faces_subdiv
        
        
        # delete unwanted area
        verts = delete_vertices(vertex_positions, verts)
        verts = rotate(rotation_angle, verts)
        
        return verts

    # If num_layers= 1 the min_rot_per_layer will be placed at the start_degree
    # If num_layers= 2 the min_rot_per_layer will be placed at start_degree and max_rot_per_layer at end_degree
    def spherical_coordinates(start_degree:float, end_degree: float, num_layers:int, min_rot_per_layer:int, max_rot_per_layer:int, z_offset:float=0.0, scale: float= 1.0):
        # takes spherical coordinate as radians and returns cartesian coordinates
        def sph2cart(az, el, r):
            rcos_theta = r * np.cos(el)
            x = rcos_theta * np.cos(az)
            y = rcos_theta * np.sin(az)
            z = r * np.sin(el)
            return (x, y, z)
        
        coordinates= []
        
        if num_layers == 1:
            for i in range(min_rot_per_layer):
                az= radians(i * (360 / min_rot_per_layer)) + radians(z_offset * (180 / min_rot_per_layer))
                el= radians(start_degree)
                
                coordinates.append(sph2cart(az, el, scale))
                
        elif num_layers == 2:
            for i in range(min_rot_per_layer):
                az= radians(i * (360 / min_rot_per_layer)) + radians(z_offset * (180 / min_rot_per_layer))
                el= radians(start_degree)
                
                coordinates.append(sph2cart(az, el, scale))
                
            for i in range(max_rot_per_layer):
                az= radians(i * (360 / max_rot_per_layer)) + radians(z_offset * (180 / max_rot_per_layer))
                el= radians(end_degree)
                
                coordinates.append(sph2cart(az, el, scale))
        
        else:
            angle_change_per_cam_layer = (abs(start_degree) + abs(end_degree)) / (num_layers-1)
            rotation_of_camera_layer = float(start_degree)
            
            
            while rotation_of_camera_layer <= end_degree:
                if start_degree >= 0:
                    z_rotations_per_render = int(np.interp(rotation_of_camera_layer, [start_degree, end_degree],
                                                            [max_rot_per_layer, min_rot_per_layer]))
                else:
                    z_rotations_per_render = int(np.interp(rotation_of_camera_layer, [start_degree, 0, end_degree],
                                                            [min_rot_per_layer, max_rot_per_layer, min_rot_per_layer]))


                for i in range(z_rotations_per_render):
                    az= radians(i * (360 / z_rotations_per_render)) + radians(z_offset * (180 / z_rotations_per_render))
                    el= radians(rotation_of_camera_layer)

                    coordinates.append(sph2cart(az, el, scale))

                rotation_of_camera_layer += angle_change_per_cam_layer

                # compensate rounding errors
                if rotation_of_camera_layer > end_degree:
                    rotation_of_camera_layer = round(rotation_of_camera_layer)
        
        return(coordinates)

    @classmethod
    def from_icosphere(cls, scale:float= 1.0, subdivisions:int= 2, rotation_angle:float=0.0, vertex_positions:str= "full_sphere"):
        args = {"scale": scale, "subdivisions": subdivisions, "rotation_angle": rotation_angle, "vertex_positions": vertex_positions}
        coords = cls.icosphere_coordinates(scale, subdivisions, rotation_angle, vertex_positions)

        return cls(coordinates= coords, method="icosphere", calculation_args=args) 

    @classmethod
    def from_spherical_coordinates(cls, start_degree:float, end_degree: float, num_layers:int, min_rot_per_layer:int, max_rot_per_layer:int, z_offset:float=0.0, scale: float= 1.0):
        args= {"start_degree":start_degree, "end_degree": end_degree, "num_layers": num_layers, "min_rot_per_layer": min_rot_per_layer, "max_rot_per_layer": max_rot_per_layer, "z_offset": z_offset, "scale": scale}
        coords = cls.spherical_coordinates(start_degree, end_degree, num_layers, min_rot_per_layer, max_rot_per_layer, z_offset, scale) 

        return cls(coordinates= coords, method="spherical coordinates", calculation_args=args)

    @classmethod
    def from_json(cls, filepath:str):
        with open(filepath, "r") as json_file:
            data_dict = json.load(json_file)

        if data_dict["type"] == "camerapath":
            return cls(method=data_dict["method"], calculation_args=data_dict["calculation_args"], coordinates=data_dict["coordinates"])
        else:
            print("Config file must be of type \"camerapath\"! Import aborted.")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list 


# Deletes all objects in the scene
def clear_scene():
    # Select all objects in scene
    bpy.ops.object.select_all(action='SELECT')

    # Delete all selected objects
    bpy.ops.object.delete()


def load_blend_file(filepath):
    bpy.ops.wm.open_mainfile(filepath=filepath)


def save_blend_file(filepath):
    bpy.ops.wm.save_as_mainfile(filepath=filepath)

# Calculates min/max values (map.inputs[...]) for depth images
def get_depth_min_max(object_name:str, cam_location:tuple, factor_min_max:tuple=(1.0, 1.0)):
    obj = bpy.data.objects[object_name]
    obj_bound_box_local_center = (1/8) * sum((Vector(o) for o in obj.bound_box), Vector())
    obj_bound_box_global_center = obj.matrix_world @ obj_bound_box_local_center

    # sphere with minimum 'radius' that entirely surrounds object's bound_box
    radius = (max(*obj.dimensions)*(2**.5))/2
    #bpy.ops.mesh.primitive_uv_sphere_add(location=obj_bound_box_global_center, radius=radius)

    # 'cam_location' distance to origin and 'obj_bound_box_global_center' distance to origin (vector length)
    cam_len = Vector(cam_location).length
    obj_len = obj_bound_box_global_center.length

    min_dist = cam_len - (obj_len + radius)
    max_dist = cam_len + (obj_len + radius)

    return (min_dist*factor_min_max[0], max_dist*factor_min_max[1])


def setup_scene(cam_resolution, format, color_depth, device="GPU", render_engine="CYCLES", compute_device_type="CUDA"):
    print(f"Setting up scene...")

    bpy.data.scenes[0].render.engine = render_engine
    print(f"    Defined {render_engine} als render engine.")

    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = compute_device_type  # or "OPENCL"
    print(f"    Defined {compute_device_type} as compute_device_type.")

    # Set the device and feature set
    bpy.context.scene.cycles.device = device
    print(f"    Set {device} as used device.")

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    print(f"    Used devices:")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(f"        ",d["name"], d["use"])
    
    scene = bpy.context.scene
    scene.render.resolution_x = cam_resolution[0]
    scene.render.resolution_y = cam_resolution[1]
    scene.render.resolution_percentage = 100
    print(f"    Set camera resolution to {cam_resolution[0]} x {cam_resolution[1]} pixels.")

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    # Set up rendering of depth map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    nodesField = bpy.context.scene.node_tree
    for currentNode in nodesField.nodes:
        nodesField.nodes.remove(currentNode)

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.render.image_settings.file_format = str(format)
    bpy.context.scene.render.image_settings.color_depth = str(color_depth)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapRange")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.

        map.inputs['From Min'].default_value = 1
        map.inputs['From Max'].default_value = 0
        map.inputs['To Min'].default_value = 1
        map.inputs['To Max'].default_value = 0

        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        normal_file_output.name = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
    print(f"    Added View Layers for normal and depth output.")

    print(f"Setting up scene ✓\n")

    return map


# Following object types can be used: MESH, CURVE, SURFACE, META, FONT, HAIR, POINTCLOUD, 
# VOLUME, GPENCIL, ARMATURE, LATTICE, EMPTY, LIGHT, LIGHT_PROBE, CAMERA, SPEAKER
def select_objects_of_type(type: str):
    bpy.ops.object.select_all(action='DESELECT')

    # Select all meshes in scene
    for object in bpy.context.scene.objects:
        if object.type == type:
            object.select_set(True)


# Deletes all selected objects
def delete_selected_objects():
    bpy.ops.object.delete()


# Add a camera to the scene
def add_camera_to_scene() -> str:
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    print(f"Added Camera with the name \"{camera_object.name}\" to Scene.\n")

    return camera_object.name


# following types can be used MESH, CURVE, SURFACE, META, FONT, HAIR, POINTCLOUD, VOLUME, 
# GPENCIL, ARMATURE, LATTICE, EMPTY, LIGHT, LIGHT_PROBE, CAMERA, SPEAKER
def get_objects_of_type(type: str = "MESH") -> list:
    objects = []
    for object in bpy.context.scene.objects:
        if object.type == type:
            objects.append(object)
    return objects


# Method to import .stl or .obj Objects
def import_model(filepath: str, use_baseplate:bool):
    if len(get_objects_of_type()) != 0:
        print("There is already a model in the scence. Deleting all models in the scene...")

        select_objects_of_type(type="MESH")
        delete_selected_objects()
        print("Deleted all models in the scene.")
    suffix = Path(filepath).suffix
    if suffix == ".stl":
        bpy.ops.import_mesh.stl(filepath=filepath)
        mesh = get_objects_of_type()
        print(f"Imported a new model with the name: {mesh[0].name}.")
    elif suffix == ".obj":
        bpy.ops.import_scene.obj(filepath=filepath)
        mesh = get_objects_of_type()
        print(f"Imported a new model with the name: {mesh[0].name}.")
    else:
        print("Wrong file formate. Use .stl or .obj files!")
        return
    
    # Move object to center
    mesh_to_center()

    if use_baseplate:
        add_base_plate()


def delete_objects_by_name(names:list):
    bpy.ops.object.select_all(action='DESELECT')

    for name in names:
        bpy.data.objects[name].select_set(True)

    bpy.ops.object.delete()


# Sets a mesh location to the center, mesh base sits on xy-axis
def mesh_to_center():
    bpy.ops.object.select_all(action='DESELECT')

    # Select model
    object = get_objects_of_type(type = "MESH")[0]
    object.select_set(True)

    # Set model origin to model center and move origin to (0,0,0)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    object.location = (0,0,0)


# Adds a base plate with size relative to the object in the scene
def add_base_plate():
    # Get object in scene
    mesh = get_objects_of_type()[0]

    # Return max value of dimensions (only evaluate xy dimension)
    xy_max = max(mesh.dimensions[:2])

    if xy_max < 5.0:
        bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, -(mesh.dimensions[2]/2)-0.0005))
    elif 5.0 <= xy_max < 7.0 :
        bpy.ops.mesh.primitive_plane_add(size=7, location=(0, 0, -(mesh.dimensions[2]/2)-0.0005))
    else:
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -(mesh.dimensions[2]/2)-0.0005))
        print("Warning! Biggest base plate is too small for the object. Consider scaling it down!")

    bpy.context.active_object.name = "base_plate"

    bpy.ops.object.select_all(action='DESELECT')

# Method to clear all world nodes
def clear_world_nodes():
    bpy.context.scene.world.node_tree.nodes.clear()


# Method to create a clean world node setup
def create_world_node_setup():
    world_nodes = bpy.context.scene.world.node_tree.nodes

    # Delete old nodes if there are any
    if len(world_nodes.items()) != 0:
        clear_world_nodes()
        print('Deleted all world nodes.')

    # Create background node
    node_background = world_nodes.new(type='ShaderNodeBackground')

    # Create environment texture node
    node_env_texture = world_nodes.new('ShaderNodeTexEnvironment')
    node_env_texture.location = -300, 0

    # Create output node
    node_output = world_nodes.new('ShaderNodeOutputWorld')
    node_output.location = 200, 0

    # Create mapping node
    node_mapping = world_nodes.new('ShaderNodeMapping')
    node_mapping.location = -500, 0

    # Create texture coordinate node
    node_texcoord = world_nodes.new('ShaderNodeTexCoord')
    node_texcoord.location = -700, 0

    # Link all nodes
    links = bpy.context.scene.world.node_tree.links
    links.new(node_texcoord.outputs['Generated'],
              node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'],
              node_env_texture.inputs['Vector'])
    links.new(node_env_texture.outputs['Color'],
              node_background.inputs['Color'])
    links.new(node_background.outputs['Background'],
              node_output.inputs['Surface'])


def create_base_plate_shader():
    mat = bpy.data.materials.new(name="base_plate_mat")
    mat.use_nodes = True

    # delete all nodes
    for node in mat.node_tree.nodes:
        mat.node_tree.nodes.remove(node)

    teximage = mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    teximage.location = -600, 0

    colorramp = mat.node_tree.nodes.new(type="ShaderNodeValToRGB")
    colorramp.location = -300,0

    huesaturation = mat.node_tree.nodes.new(type="ShaderNodeHueSaturation")
    huesaturation.location = 0,0

    gamma = mat.node_tree.nodes.new(type="ShaderNodeGamma")
    gamma.location = 200, 0

    diffuse = mat.node_tree.nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.location = 400, 0

    out = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    out.location = 600, 0

    links = mat.node_tree.links
    links.new(teximage.outputs["Color"], colorramp.inputs["Fac"])
    links.new(colorramp.outputs["Color"], huesaturation.inputs["Color"])
    links.new(huesaturation.outputs["Color"], gamma.inputs["Color"])
    links.new(gamma.outputs["Color"], diffuse.inputs["Color"])
    links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])

    bpy.data.objects["base_plate"].data.materials.append(mat)
    print("Created shader node setup for base_plate.")


# Method to set a background image, creates node setup if necessary
def set_background_image(filepath: str):
    # Add world node setup if environment texture node is not in world nodes
    if len([item for item in bpy.context.scene.world.node_tree.nodes.items() if 'Environment Texture' in item]) == 0:
        print("No environment node in world nodes! Creating new world node setup...")
        create_world_node_setup()
        print("Created new world node setup.\n")

    # Add 360 degree image
    bpy.data.worlds['World'].node_tree.nodes['Environment Texture'].image = bpy.data.images.load(
        filepath)
    print(f"Set image: {Path(filepath).name} as background image.\n")


# Method to load materials from an other .blend file. If is_link is true, the materials are only linked to the current file
def load_material(filepath: str, materials: list, is_link: bool = False):
    with bpy.data.libraries.load(filepath, link=is_link) as (_, data_to):
        data_to.materials = materials
        print(f"Loaded material(s): {materials} from {Path(filepath).name}")

    # Activate Fake User for every Material (If False all unused materials will be deleted when closing the file.)
    for m in materials:
        m.use_fake_user = True


# Returns a list of all available Materials in a .blend file
def list_materials_of_blend_file(filepath: str) -> list:
    with bpy.data.libraries.load(filepath, link=True) as (data_from, _):
        return data_from.materials


# Append a material to an object
def append_material_to_object(object_name: str, material_name: str):
    bpy.data.objects[object_name].data.materials.clear()
    bpy.data.objects[object_name].data.materials.append(
        bpy.data.materials[material_name])
    print(f"Applied material: \"{material_name}\" to object: \"{object_name}\".")


# Removes shaders from the baseplate and Object and appends an generic BSDF
def remove_shaders(object_name: str, use_base_plate: bool):
    obj = bpy.data.objects[object_name]

    if use_base_plate:
        plane = bpy.data.objects["base_plate"]
        mat_new_plane = bpy.data.materials.new(name="plane grey surface")
        mat_new_plane.use_nodes = True
        plane_bsdf = mat_new_plane.node_tree.nodes.get("Principled BSDF")
        plane_bsdf.inputs['Base Color'].default_value = (.8, .8, .8, 1.0)
        plane.data.materials.clear()
        plane.data.materials.append(mat_new_plane)

    # Create new material with 'Principled BSDF'
    mat_new_part = bpy.data.materials.new(name="obj grey surface")
    mat_new_part.use_nodes = True

    # Specify color of base plate and object (RGBA)
    part_bsdf = mat_new_part.node_tree.nodes.get("Principled BSDF")
    part_bsdf.inputs['Base Color'].default_value = (.333, .333, .333, 1.0)

    # Remove all existing materials and add newly created ones
    obj.data.materials.clear()
    obj.data.materials.append(mat_new_part)


# Creates a camera and a sets a track-to constraint to an empty at the center, return camera name
def initialize_camera() -> str:
    # Delete old cameras
    select_objects_of_type("CAMERA")
    delete_selected_objects()
    print("Deleted old camera(s).")
    
    # Add new camera
    cam = add_camera_to_scene()
    
    # Add empty at center to track camera to
    track_to_empty = add_empty_at_location(location=(0,0,0), name="track_to_empty")
    
    # Track camera to empty
    ttc= bpy.data.objects[cam].constraints.new(type='TRACK_TO')
    ttc.target = bpy.data.objects[track_to_empty]

    return cam


def camera_scale_to_object(object_name:str, scale_factor:float=1.02) -> float:
    # Crude intervall nesting method to calculate the tangent intersection with the boundix box sphere
    def _tangent_intersection(gradient:float, radius:float, steps:int = 1000, tolerance:float= 0.0001) -> tuple:
        # Quarter circle function
        f = lambda x,r: (r**2 - (x - r)**2)**0.5
        
        # Derivative of a quarter circle function
        f_p = lambda x,r : - (x - r) / (r**2 - (x - r)**2)**0.5
       
        x_max = radius
        x_min = 0.0
        x_t = 0.0
        
        for _ in range(steps):
            x_t = (x_max + x_min) / 2
            f_t = f_p(x_t, radius)

            if abs(f_t - gradient) <= tolerance: break
            
            if f_t > gradient:
                x_min = x_t
            elif f_t < gradient:
                    x_max = x_t
            else:
                break
        
        # Calcualte f(x) value for point x_t
        f_t = f(x_t, radius)
        
        return (x_t, f_t)


    # Horizontal and vertical camera view angle
    render_width = bpy.context.scene.render.resolution_x
    render_height = bpy.context.scene.render.resolution_y
    aspect = render_width / render_height
    cam = get_objects_of_type(type= "CAMERA")[0]

    if aspect > 1:
        hFOV = cam.data.angle
        vFOV = 2 * atan((0.5 * render_height) / (0.5 * render_width / atan(hFOV / 2)))
    else:
        vFOV = cam.data.angle
        hFOV = 2 * atan((0.5 * render_width) / (0.5 * render_height / atan(vFOV / 2)))


    h_angle = hFOV / 2
    v_angle = vFOV / 2
    
    # Calcualte gradients
    h_grad = tan(h_angle)
    v_grad = tan(v_angle)
    
    # Object data
    obj = bpy.data.objects[object_name]
    
    # Calculate length of the inner diagonal
    inner_diagonal = (obj.dimensions[0]**2 + obj.dimensions[1]**2 + obj.dimensions[2]**2)**0.5
    
    # Radius of a sphere that includes all vertices of the bounding box
    radius = inner_diagonal / 2
        
    x_intersect = _tangent_intersection(h_grad, radius)
    y_intersect = _tangent_intersection(v_grad, radius)
    
    h_cam_radius = abs(x_intersect[1] / h_grad) + x_intersect[0]
    v_cam_radius = abs(y_intersect[1] / v_grad) + y_intersect[0]
        
    return max(h_cam_radius, v_cam_radius) * scale_factor


# Sets a camera to location, camera=camera name
def set_camera_location(camera: str, location: tuple):
    cam_object = bpy.data.objects[camera]
    cam_object.location=location

    print(f"Set {camera} at location: x: {location[0]}, y: {location[1]}, z: {location[2]}")


# add an empty at a location and returns name of the empty
def add_empty_at_location(location: tuple, name:str= "empty", size:float = 0.5) -> str:
    o = bpy.data.objects.new( name, None )

    bpy.context.scene.collection.objects.link( o )

    o.location=location
    o.empty_display_size = size
    o.empty_display_type = 'PLAIN_AXES' 

    return o.name 
