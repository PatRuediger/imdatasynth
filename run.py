import argparse
import sys
from blenderFBK import *
import json
import pandas as pd
from pathlib import Path


def main(run_config):
    # Check if all config files can be accessed
    global config, working_directory
    config, working_directory, datasets = _consistency_check(run_config)

    global lightsetup 
    lightsetup= LightSetup()

    # Delete all cameras and add a new one with tracking constraints
    initialize_camera()

    # Define render settings
    global depth_node
    depth_node = setup_scene(cam_resolution=config["cam_resolution"], format=config["file_format"], color_depth=config["color_depth"], device=config["device"], compute_device_type=config["compute_device_type"])

    # Create dataframe for csv data
    global cols
    cols = ["set", "label", "category_1", "category_2", "category_3", "labelID", "variantNum", "attributes", "dataset", "baseResolution", "prefix", "filename"] 
    csv_data = pd.DataFrame(columns=cols)

    # Run per dataset
    for dataset in datasets:
        print(f"Rendering {dataset} dataset...")
        csv_data = _per_dataset(dataset= dataset, working_directory= working_directory, csv_data= csv_data)

    if not working_directory.joinpath(f"data_complete.csv").exists():
        # Save csv data
        csv_data.to_csv(working_directory.joinpath(f"data_complete.csv"), sep=";", index=False)
    else:
        return

def _consistency_check(run_config):
    print(f"Checking blend file and config...")
    with open(run_config, "r") as json_file:
        config = json.load(json_file)

        working_directory = Path(run_config).parent

        datasets = config["datasets"]

        print(f"    Loaded config file ✓")
    
    # Check if all files are present
    for dataset in datasets:
        for file in datasets[dataset]["files"].values():
            assert working_directory.joinpath(file).is_file(), f"Error: Couldn't find file: {working_directory.joinpath(file)}"
    print(f"    All files in config are accessible ✓")

    # Get all names of objects
    objects = [x.name for x in get_objects_of_type(type= "MESH")]
    
    # Check if object is in scene
    assert config["object_name"] in objects, f"Error: Object is not in the scene."
    print(f"    Model is in the scene ✓")

    # Check if base_plate is in scene
    if config["use_base_plate"]:
        assert "base_plate" in objects, "Error: No base_plate in the scene."
        print(f"    Base_plate is in the scene ✓")

    # Check if all materials are in the blend file
    avail_mats = [x.name for x in bpy.data.materials]
    mat = "material" # To use in formatted string
    bmat = "base_plate_mat" # To use in formatted string

    for dataset in datasets:
        assert datasets[dataset][mat] in avail_mats, f"Error: Material \"{datasets[dataset][mat]}\" for dataset \"{dataset}\" is missing in blend file!"
        assert datasets[dataset][bmat] in avail_mats, f"Error: Material \"{datasets[dataset][bmat]}\" for dataset \"{dataset}\" is missing in blend file!"
    del mat, bmat
    print(f"    All required materials are in the blend file ✓")

    print(f"Consistency check raised no error.\n")

    return config, working_directory, datasets


def _per_dataset(dataset: str, working_directory:Path, csv_data):
    dataset_config = config["datasets"][dataset]
    files = dataset_config["files"]

    # Set background image
    set_background_image(filepath=str(working_directory.joinpath(files["background"])))
    
    # Delete all lights and load lightsetup
    lightsetup.clear_scene()
    lightsetup.from_json(filepath=str(working_directory.joinpath(files["lightsetup"])))
    lightsetup.create()
    print("\n")

    # Camerapath
    camerapath = CameraPath.from_json(filepath=str(working_directory.joinpath(files["camera_config"])))

    # Adjust depth min max
    depth_min_max = get_depth_min_max(object_name= config["object_name"], cam_location=camerapath.coordinates[0])
    depth_node.inputs['From Min'].default_value = depth_min_max[0]
    depth_node.inputs['From Max'].default_value = depth_min_max[1]
    print(f"Calculated and applied depth values.\n")

    # Attribute to use in csv file
    global attr

    # Category folder
    cf = working_directory.joinpath(dataset, config["categories"][1], config["variantNum"], config["categories"][0])

    # Create shaders_with_bg data
    if "shaders_with_bg" in config["attributes"]:
        attr = "shaders_with_bg"
        print(f"Crreating {attr} data.")

        # Set backbround not transparent.
        bpy.context.scene.render.film_transparent = False

        # Apply material to object and base_plate
        append_material_to_object(object_name=config["object_name"], material_name=dataset_config["material"])
        if config["use_base_plate"]:
            append_material_to_object(object_name="base_plate", material_name= dataset_config["base_plate_mat"])
        print(f"\n")

        out = cf.joinpath(attr, str(config["cam_resolution"][0]) + "_" + str(config["cam_resolution"][1]))
        out.mkdir(parents= True, exist_ok= True)

        csv_data = _render(dir= out, coordinates= camerapath.coordinates, dataset= dataset, csv_data=csv_data)

    # Create shaders_with_bg data
    if "shaders_without_bg" in config["attributes"]:
        attr = "shaders_without_bg"
        print(f"Creating {attr} data.")

        # Set backbround transparent.
        bpy.context.scene.render.film_transparent = True

        # Apply material to object
        append_material_to_object(object_name=config["object_name"], material_name=dataset_config["material"])
        if config["use_base_plate"]:
            append_material_to_object(object_name="base_plate", material_name= dataset_config["base_plate_mat"])
        print(f"\n")

        out = cf.joinpath(attr, str(config["cam_resolution"][0]) + "_" + str(config["cam_resolution"][1]))
        out.mkdir(parents= True, exist_ok= True)

        csv_data = _render(dir= out, coordinates= camerapath.coordinates, dataset= dataset, csv_data=csv_data)

     # Create without_shaders data
    if "without_shaders" in config["attributes"]:
        attr = "without_shaders"
        print(f"Creating {attr} data.")

        remove_shaders(object_name=config["object_name"], use_base_plate= config["use_base_plate"])

        # Set backbround transparent.
        bpy.context.scene.render.film_transparent = False

        out = cf.joinpath(attr, str(config["cam_resolution"][0]) + "_" + str(config["cam_resolution"][1]))
        out.mkdir(parents= True, exist_ok= True)

        csv_data = _render(dir= out, coordinates= camerapath.coordinates, dataset= dataset, csv_data=csv_data)

       

    return csv_data

        
def _render(dir, coordinates, dataset, csv_data):
    # Data to store in JSON file
    transform_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }
    transform_data['frames'] = []

    for (i, coord) in enumerate(coordinates[:5]):
        set_camera_location("Camera", coord)

        bpy.context.scene.render.filepath =   str(dir) + '/r_' + str(i).zfill(3)
        fp = Path(bpy.context.scene.render.filepath)

        if config["create_depth_normal"]:
            bpy.context.scene.node_tree.nodes['Depth Output'].base_path = ""
            bpy.context.scene.node_tree.nodes['Normal Output'].base_path = ""
            bpy.context.scene.node_tree.nodes['Depth Output'].file_slots[0].path =  str(dir) + '/rd_' + str(i).zfill(3) + "_depth_"
            bpy.context.scene.node_tree.nodes['Normal Output'].file_slots[0].path =  str(dir) + '/rn_' + str(i).zfill(3) + "_normal_"

        bpy.ops.render.render(write_still=True)
        
        frame_data = {
            'file_path': "./" + dataset + "/" + config["categories"][1] + "/" + config["variantNum"] + "/" + config["categories"][0] + '/r_' + str(i).zfill(3),
            'transform_matrix': listify_matrix(bpy.data.objects["Camera"].matrix_world)
        }
        transform_data['frames'].append(frame_data)

        frame_csv = pd.DataFrame([[dataset, config["object_name"], config["categories"][0], config["categories"][1], config["categories"][2], config["labelID"], config["variantNum"], attr, config["source"], str(config["cam_resolution"][0]) + "_" + str(config["cam_resolution"][1]), fp.parent, fp.name]], columns=cols)
        csv_data = csv_data.append(frame_csv)
        
    with open(working_directory.joinpath(f"transforms_{dataset}.json"), 'w') as out_file:
            json.dump(transform_data, out_file, indent=4)

    return csv_data


if __name__ == "__main__":
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:] # all arguments after ' -- '
        parser = argparse.ArgumentParser()
        parser.add_argument('-cfg', '--config', type=str, required=True)
        args = parser.parse_known_args(argv)[0]
        main(args.config)