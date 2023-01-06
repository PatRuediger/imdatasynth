from ipywidgets import *
from ipyfilechooser import FileChooser
from widgets import ExperimentLoader, DatasetChooser, Options, GeneralTab, ModelTab, MaterialTab, BackgroundTab, CameraviewTab, LightingTab, DatasetChecker, Save_Run_Experiment
from pathlib import Path
from blenderFBK import *
import numpy as np
from math import sqrt, radians, tan
import mathutils
from mathutils import Vector
import PIL.Image
import io
import shutil
import run
import copy


import json
import bpy

class Main:
    def __init__(self) -> None:
        clear_scene()
        initialize_camera()

        self.lightsetup = LightSetup()

        self.config = {}
        with open("empty_config.json", "r") as json_file:
            self.config = json.load(json_file)

        self.dataset_config = {
                    "files": {
                        "background": "",
                        "camera_config": "",
                        "lightsetup": ""
                            },
                    "material": "",
                    "base_plate_mat": ""
                    }

        self.camerapaths = {}
        self.lightconfigs = {}
        self.background_images = {}

        self.test = self.config

        self.info_label = HTML()
        self._LBL_TEMPLATE = '<span style="color:{1};">{0}</span>'

        self._has_model_state = Checkbox(value=False)
        self._experiment_complete_state = Checkbox(value=False)

        self.el = ExperimentLoader()
        el_context = self.el.instantiate()
        self.el_box = HBox([el_context, self.info_label])
        display(self.el_box)

        self.dc = DatasetChooser()
        self.o = Options()
        self.gt = GeneralTab()
        self.mt = ModelTab()
        self.bt = BackgroundTab()
        self.matt = MaterialTab()
        self.ct = CameraviewTab()
        self.lt = LightingTab()
        self.datasetchecker = DatasetChecker()
        self.sr = Save_Run_Experiment()

        # General Options
        self.object_name = Text(description="Object name:", placeholder="Name of the object")

        self.el.load_btn.on_click(self._load_btn_click)
        self.dc.add_btn.on_click(self._add_dataset_btn_click)
        self.dc.del_btn.on_click(self._del_dataset_btn_click)
        self.gt.apply_general_btn.on_click(self._apply_general)
        self.mt.apply_model_btn.on_click(self._apply_model)

        # Check if general is applied and scene has model then allow scale calculation
        self.gt.applied_state.observe(self._allow_scale_calculation, "value")
        self._has_model_state.observe(self._allow_scale_calculation, "value")

        self.dc.datasets.observe(self._on_dataset_change, "value")
        

        # Observations for apply button of model tab
        self._has_model_state.observe(self._is_model_filled, "value")
        self.mt.object_name.observe(self._is_model_filled, "value")
        self.mt.labelID.observe(self._is_model_filled, "value")
        self.mt.fc._select.observe(self._is_model_filled, "description")

        # Material Tab
        self.matt.load_database_btn.on_click(self._on_load_material_database_click)
        self.mt.baseplate.observe(self._on_use_baseplate_change, "value")
        self.matt.apply_btn.on_click(self._on_apply_material_btn_click)

        # Background Tab
        self.bt.apply_btn.on_click(self._on_background_apply_btn_click)

        # Cameraview tab
        self.ct.calculate_btn.on_click(self._cameraview_calculate_btn_click)
        self.ct.visualize.observe(self._visualize_camerapath, "value")
        self.ct.calc_scale.observe(self._on_calc_scale_change, "value")
        self.ct.vis_slider.observe(self.on_vis_slider_change, "value")
        self.ct.save_btn.on_click(self._on_save_camerapath_btn_click)
        self.ct.load_btn.on_click(self._on_load_camerapath_btn_click)

        # Lightsetup tab
        self.lt.update_btn.on_click(self._on_update_lightsetup_btn_click)
        self.lt.visualize.observe(self._on_lightsetup_visualize_change, "value")
        self.lt.load_btn.on_click(self._on_load_lightconfig_btn_click)
        self.lt.save_btn.on_click(self._on_save_lightconfig_save_btn_click)

        # Save Run experiment
        self.sr.save_btn.on_click(self._on_save_experiment_btn_click)
        self.sr.run_btn.on_click(self._on_run_experiment_btn_click)

        self._experiment_complete_state.observe(self._experiment_state_change, "value")
        self.sr.fc._select.observe(self._on_sr_label_change, "description")


    def _experiment_state_change(self, change):
        if change.new:
            self.sr.save_btn.disabled=False
            self.sr.run_btn.disabled=False
        else:
            self.sr.save_btn.disabled=True
            self.sr.run_btn.disabled=True


    def _load_btn_click(self, btn):
        if self.el._consistency_check():
            self.info_label.value = self._LBL_TEMPLATE.format("Folder is consistent.", 'green')

            self._load_experiment()            
        else:
            self.info_label.value = self._LBL_TEMPLATE.format("Unable to load experiment. Folder not consistent.", 'red')


    def _add_dataset_btn_click(self, btn):
        if self.dc.textfield.value not in self.dc.datasets.options and not str(self.dc.textfield.value).isspace():
            self.dc.datasets.options = self.dc.datasets.options + (self.dc.textfield.value,)
            
            self.config["datasets"].update({self.dc.textfield.value: copy.deepcopy(self.dataset_config)})

            self.camerapaths.update({self.dc.textfield.value: {}})
            self.lightconfigs.update({self.dc.textfield.value: {}})
            self.background_images.update({self.dc.textfield.value: {}})

            self.dc.textfield.value = ""

            self._experiment_complete_state.value = self._completeness_check(self.config)

        
    def _del_dataset_btn_click(self, btn):
        tmp_list = list(self.dc.datasets.options)

        self.config["datasets"].pop(self.dc.datasets.value)

        self.camerapaths.pop(self.dc.datasets.value)

        self.lightconfigs.pop(self.dc.datasets.value)

        self.background_images.pop(self.dc.datasets.value)

        tmp_list.remove(self.dc.datasets.value)
        self.dc.datasets.options = tmp_list

        self._experiment_complete_state.value = self._completeness_check(self.config)

    def _load_experiment(self):
        self.info_label.value = self._LBL_TEMPLATE.format("Loading experiment...", 'orange')

        global files, dir
        self.config = {}
        dir = Path(self.el.fc.selected)
        blend_file = [x for x in dir.glob("*.blend")][0]
        json_files = [x for x in dir.glob("*.json")]
            
        load_blend_file(str(blend_file))

        # Initialize material tab with exisitng materials
        init_mats = [x.name for x in bpy.data.materials if x.name != "Dots Stroke"]
        self.matt.material_model.options=init_mats
        self.matt.material_base_plate.options=init_mats

        self._scene_has_object()
        
        # Find the right config file
        for j in json_files:
            with open(j, "r") as json_file:
                cfg = json.load(json_file)
                try:
                    cfg["object_name"]
                    
                    self.config = cfg
                    break
                except:
                    continue
                    
        files = dir.joinpath("configs")


        # Collect camerapaths from existing datasets
        camerapath_dirs = {}
        self.camerapaths={}
        for key in self.config["datasets"]:
            camerapath_dirs.update({key: self.config["datasets"][key]["files"]["camera_config"]})

        for key in camerapath_dirs:
            f = dir.joinpath(camerapath_dirs[key])

            with open(f, "r") as json_file:
                cam_cfg = json.load(json_file)

                self.camerapaths.update({key: cam_cfg})
        
        # Collect lightconfigs from existing datasets
        lightconfig_dirs = {}
        self.lightconfigs={}
        for key in self.config["datasets"]:
            lightconfig_dirs.update({key: self.config["datasets"][key]["files"]["lightsetup"]})

        for key in lightconfig_dirs:
            f = dir.joinpath(lightconfig_dirs[key])

            with open(f, "r") as json_file:
                light_cfg = json.load(json_file)

                self.lightconfigs.update({key: light_cfg})

        # Collect background images from existing datasets
        self.background_images = {}
        for key in self.config["datasets"]:
            self.background_images.update({key: str(dir.joinpath(self.config["datasets"][key]["files"]["background"]))})


        # Add datasets to DatasetChooser
        datasets = [x for x in self.config["datasets"]]
        self.dc.datasets.options = datasets

        self._load_general_from_config()
        self._load_model_from_config()

        # Set Directory of Save and Reload widget
        self.sr.fc.reset(path=Path(self.el.fc.selected).parent, filename=Path(self.el.fc.selected).name)
        self.sr.fc._apply_selection()

        # Add camera if there is none, if there is create new tracking constraint
        if len(get_objects_of_type(type="CAMERA")) == 0:
            initialize_camera()
        else:
            cam = get_objects_of_type(type="CAMERA")[0]
            track_to_empty = add_empty_at_location(location=(0,0,0), name="track_to_empty")
        
            ttc= bpy.data.objects[cam.name].constraints.new(type='TRACK_TO')
            ttc.target = bpy.data.objects[track_to_empty]
        
        
        if all(self.config):
            self.sr.save_btn.disabled=False
            self.sr.run_btn.disabled=False

        self.info_label.value = self._LBL_TEMPLATE.format("Experiment loaded.", 'green')


    def _scene_has_object(self):
        scene_objects = [x.name for x in get_objects_of_type("MESH") if x.name != "base_plate"]

        if len(scene_objects) > 0:
            self._has_model_state.value = True
        else:
            self._has_model_state.value = False


    def _is_model_filled(self, _):
        fields = [self.mt.object_name.value, self.mt.labelID.value, any([self._has_model_state.value, self.mt._model_selected_state.value])]
        if all(fields):
            self.mt.apply_model_btn.disabled = False
        else:
            self.mt.apply_model_btn.disabled = True


    def _on_dataset_change(self, change):
        if change.new == None:
            children = [self.o._empty_dataset()] * len(self.o.children)
            self.o.tab.children = children

            self.lightsetup.data = {}
            with open("empty_config.json", "r") as json_file:
                self.config = json.load(json_file)

        else:
            self.info_label.value = self._LBL_TEMPLATE.format("Loading Dataset...", "orange")
            children = [self.gt._return_context(), self.mt._return_context(), self.matt.return_context(), self.bt.return_context(), self.ct._return_context(), self.lt.return_context()]

            self.o.tab.children = children

            self._update_cameraview_tab(change)
            self._update_lightsetup_tab(change)
            self._update_material_tab(change)
            self._update_background_tab(change)

            # Update Statusbar
            # When adding the first dataset to an empty experiment the check can be triggered before the config is adjusted -> only run check if config is already extended
            if self.dc.datasets.value in self.config["datasets"]:
                self.datasetchecker.update_statusbar( 
                    self.datasetchecker.is_generel_applied(self.config),
                    self.datasetchecker.is_model_applied(self.config["object_name"], self.config["labelID"]),
                    self.datasetchecker.is_material_applied(self.config["datasets"][self.dc.datasets.value]["material"], self.config["datasets"][self.dc.datasets.value]["base_plate_mat"]),
                    self.datasetchecker.is_background_applied(self.config["datasets"][self.dc.datasets.value]["files"]["background"]),
                    self.datasetchecker.is_cameraviews_applied(self.config["datasets"][self.dc.datasets.value]["files"]["camera_config"]),
                    self.datasetchecker.is_lighting_applied(self.config["datasets"][self.dc.datasets.value]["files"]["lightsetup"]),
                    self.datasetchecker.is_exp_dir_set(self.sr.fc.selected)
                )

            self.info_label.value = self._LBL_TEMPLATE.format("Dataset loaded.", "green")


    def _load_general_from_config(self):
        self.gt.source.value = self.config["source"]
        self.gt.x_resolution.value = str(self.config["cam_resolution"][0])
        self.gt.y_resolution.value = str(self.config["cam_resolution"][1])
        self.gt.file_format.value = self.config["file_format"]
        self.gt.color_depth.value = self.config["color_depth"]
        self.gt.variant_num.value = self.config["variantNum"]
        self.gt.category1.value = self.config["categories"][0]
        self.gt.category2.value = self.config["categories"][1]
        self.gt.category3.value = self.config["categories"][2]

        if "without_shaders" in self.config["attributes"]:
            self.gt.without_shaders.value = True
        else:
            self.gt.without_shaders.value = False

        if "shaders_with_bg" in self.config["attributes"]:
            self.gt.shaders_with_bg.value = True
        else:
            self.gt.shaders_with_bg.value = False

        if "shaders_without_bg" in self.config["attributes"]:
            self.gt.shaders_without_bg.value = True
        else:
            self.gt.shaders_without_bg.value = False

        self.gt.create_depth_normal.value = self.config["create_depth_normal"]
        self.gt.device.value = self.config["device"]
        self.gt.compute.value = self.config["compute_device_type"]


    def _load_model_from_config(self):
        self.mt.object_name.value = self.config["object_name"]
        self.mt.labelID.value = self.config["labelID"]
        self.mt.baseplate.value = self.config["use_base_plate"]


    def _apply_general(self, btn):
        self.config["source"] = self.gt.source.value
        self.config["cam_resolution"] = [int(self.gt.x_resolution.value), self.gt.y_resolution.value]
        self.config["file_format"] = self.gt.file_format.value
        self.config["color_depth"] = int(self.gt.color_depth.value)
        self.config["categories"] = [self.gt.category1.value, self.gt.category2.value, self.gt.category3.value]
        self.config["variantNum"] = self.gt.variant_num.value
        self.config["create_depth_normal"] = self.gt.create_depth_normal.value
        self.config["device"] = self.gt.device.value
        self.config["compute_device_type"] = self.gt.compute.value

        tmp_attr = []
        if self.gt.without_shaders.value:
            tmp_attr.append("without_shaders")

        if self.gt.shaders_with_bg.value:
            tmp_attr.append("shaders_with_bg")

        if self.gt.shaders_without_bg.value:
            tmp_attr.append("shaders_without_bg")

        self.config["attributes"] = tmp_attr

        # Apply camera resolution
        scene = bpy.context.scene
        scene.render.resolution_x = int(self.gt.x_resolution.value)
        scene.render.resolution_y = self.gt.y_resolution.value

        # Check if general is successfully applied and change dataset status
        self.gt.applied_state.value = self.datasetchecker.is_generel_applied(self.config)
        self.datasetchecker.set_general_label_status(self.gt.applied_state.value)

        self._experiment_complete_state.value = self._completeness_check(self.config)

        self.info_label.value = self._LBL_TEMPLATE.format("Applied general configuration.", 'green')

    def _apply_model(self, btn):

        self.config["object_name"] = self.mt.object_name.value
        self.config["labelID"] = self.mt.labelID.value
        self.config["use_base_plate"] = self.mt.baseplate.value


        try:
                delete_objects_by_name(names=["base_plate"])
        except:
            pass

        if self.mt._model_selected_state.value:
            import_model(filepath=self.mt.fc.selected, use_baseplate=self.mt.baseplate.value)
            self._has_model_state.value = True

        if self.mt.baseplate.value:
            if "base_plate" in [x.name for x in get_objects_of_type(type="MESH")]:
                delete_objects_by_name(names=["base_plate"])
            add_base_plate()


        model_name = [x.name for x in get_objects_of_type("MESH") if x.name != "base_plate"][0]
        bpy.data.objects[model_name].name = self.mt.object_name.value

        # Update Statusbar and check experiment completeness
        self.datasetchecker.set_model_label_status(self.datasetchecker.is_model_applied(self.config["object_name"], self.config["labelID"]))
        self._experiment_complete_state.value = self._completeness_check(self.config)

        self.info_label.value = self._LBL_TEMPLATE.format("Applied model configuration.", 'green')
        

    def _cameraview_calculate_btn_click(self, btn):
        if self.ct.method.value == "Spherical Coordinates":
            camerapath = CameraPath.from_spherical_coordinates(start_degree=self.ct.sph_degree_range.value[0], end_degree=self.ct.sph_degree_range.value[1], 
                        num_layers=self.ct.num_layers.value, min_rot_per_layer=self.ct.sph_rotations.value[0], 
                        max_rot_per_layer=self.ct.sph_rotations.value[1], 
                        z_offset=self.ct.sph_offset.value, scale=self.ct.scale.value)
        
            self.camerapaths[self.dc.datasets.value] = camerapath.__dict__

        elif self.ct.method.value == "Icosphere":
            camerapath = CameraPath.from_icosphere(scale=self.ct.scale.value, subdivisions=self.ct.subdiv.value, rotation_angle=self.ct.rotation.value, 
                        vertex_positions= self.ct.ico_options.value)

            self.camerapaths[self.dc.datasets.value] = camerapath.__dict__

        else:
            if len(self.ct.cc.cw.all_x_coords) > 0:
                coordinates = []
                for i in range(len(self.ct.cc.cw.all_x_coords)):
                    coordinates.append([self.ct.cc.cw.all_x_coords[i], self.ct.cc.cw.all_y_coords[i], self.ct.cc.cw.all_z_coords[i]])
                
                camerapath = CameraPath(coordinates=coordinates, method="custom", calculation_args={})
                self.camerapaths[self.dc.datasets.value] = camerapath.__dict__


        self.ct.visualize.disabled = False

        if self.ct.visualize.value == True:
            emptys = [x.name for x in bpy.data.objects if "camera coordinate" in x.name]
            delete_objects_by_name(names=emptys)

            coords = self.camerapaths[self.dc.datasets.value]["coordinates"]
            
            self.ct.vis_slider.max = len(coords)-1
            
            cam = [x.name for x in get_objects_of_type(type="CAMERA")][0]
            set_camera_location(camera=cam, location=coords[self.ct.vis_slider.value])

            for coord in coords:
                add_empty_at_location(location=coord, name="camera coordinate")

        # Add name to config if empty
        if not self.config["datasets"][self.dc.datasets.value]["files"]["camera_config"]:
            self.config["datasets"][self.dc.datasets.value]["files"]["camera_config"] = f"configs/{self.dc.datasets.value}_camerapath.json"

        # Update Statusbar and check experiment completeness
        self.datasetchecker.set_cameraviews_label_status(self.datasetchecker.is_cameraviews_applied(self.config["datasets"][self.dc.datasets.value]["files"]["camera_config"]))
        self._experiment_complete_state.value = self._completeness_check(self.config)

        self.info_label.value = self._LBL_TEMPLATE.format(f"Applied camerapath configuration to {self.dc.datasets.value} dataset.", 'green')


    def _visualize_camerapath(self, change):
        if change.new:
            coords = self.camerapaths[self.dc.datasets.value]["coordinates"]

            self.ct.vis_slider.max = len(coords)-1
            self.ct.vis_slider.disabled= False

            for coord in coords:
                add_empty_at_location(location=coord, name="camera coordinate")
        else:
            emptys = [x.name for x in bpy.data.objects if "camera coordinate" in x.name]
            delete_objects_by_name(names=emptys)

            self.ct.vis_slider.disabled = True

    def _on_calc_scale_change(self, change):
            if change.new:
                self.ct.scale.value = camera_scale_to_object(object_name=self.config["object_name"])
                self.ct.scale.disabled = True
            else:
                self.ct.scale.disabled = False


    def on_vis_slider_change(self, change):
        cam = [x.name for x in get_objects_of_type(type="CAMERA")][0]
        set_camera_location(camera=cam, location=self.camerapaths[self.dc.datasets.value]["coordinates"][int(change.new)])


    def _update_material_tab(self, change):
        try:
            self.matt.material_model.value = self.config["datasets"][self.dc.datasets.value]["material"]
            self.matt.material_base_plate.value = self.config["datasets"][self.dc.datasets.value]["base_plate_mat"]
        except:
            pass


    def _update_background_tab(self, change):
        try:
            p = Path(self.background_images[self.dc.datasets.value])
            parent = p.parent
            fname = p.name
            self.bt.fc.reset(path=parent, filename=fname)
            self.bt.fc._apply_selection()

            with PIL.Image.open(self.background_images[self.dc.datasets.value]) as image:
                img_byte_arr = io.BytesIO()
                height_percent = (self.bt.preview_height*2 / float(image.size[1]))
                width_size = int((float(image.size[0]) * float(height_percent)))
                scaled_image = image.resize((width_size, self.bt.preview_height*2), PIL.Image.NEAREST)  
                scaled_image.save(img_byte_arr, format=image.format)
                self.bt.background_preview.value = img_byte_arr.getvalue()
        except:
            pass


    def _update_cameraview_tab(self, change):
        # Try to find existing file
        try:
            file = Path(self.config["datasets"][self.dc.datasets.value]["files"]["camera_config"])
            tmp_dir = dir.joinpath(*file.parents)

            self.ct.fc.reset(path=tmp_dir, filename=file.name)
            self.ct.fc._apply_selection()
        except:
            pass

        # Try loading existing config
        try:
            tmp_camerapath = self.camerapaths[self.dc.datasets.value]
            calc_args = tmp_camerapath["calculation_args"]

            if len(tmp_camerapath) > 0:
                if tmp_camerapath["method"] == "spherical coordinates":
                    self.ct.method.value = "Spherical Coordinates"
                    self.ct.sph_degree_range.value = [calc_args["start_degree"], calc_args["end_degree"]]
                    self.ct.num_layers.value = calc_args["num_layers"]
                    self.ct.sph_rotations.value = [calc_args["min_rot_per_layer"], calc_args["max_rot_per_layer"]]
                    self.ct.scale.value = calc_args["scale"]
                    self.ct.sph_offset.value = calc_args["z_offset"]
                elif tmp_camerapath["method"] == "icosphere":
                    self.ct.method.value = "Icosphere"
                    self.ct.scale.value = calc_args["scale"]
                    self.ct.subdiv.value = calc_args["subdivisions"]
                    self.ct.rotation.value = calc_args["rotation_angle"]
                    self.ct.ico_options.value = calc_args["vertex_positions"]
                else:
                    self.ct.method.value = "Custom"
                    self.ct.cc.cw.all_x_coords = [x[0] for x in tmp_camerapath["coordinates"]]
                    self.ct.cc.cw.all_y_coords = [x[1] for x in tmp_camerapath["coordinates"]]
                    self.ct.cc.cw.all_z_coords = [x[2] for x in tmp_camerapath["coordinates"]]
                    self.ct.cc.cw.instantiate()

                    # initially activate delete button if there are values
                    if len(self.ct.cc.cw.all_x_coords) > 0:
                        self.ct.cc.del_button.disabled = False
        except:
            pass

        self.ct._instantiate(method=self.ct.method.value)

    def _update_lightsetup_tab(self, change):
        # Try to find existing file
        try:
            file = Path(self.config["datasets"][self.dc.datasets.value]["files"]["lightsetup"])
            tmp_dir = dir.joinpath(*file.parents)

            self.lt.fc.reset(path=tmp_dir, filename=file.name)
            self.lt.fc._apply_selection()
        except:
            pass
        
        try:
            self.lightsetup.clear_scene()
            self.lightsetup.from_dict(self.lightconfigs[self.dc.datasets.value])

            if self.lt.visualize.value:
                self.lightsetup.create()
        except:
            pass


    def _allow_scale_calculation(self, change):
        if self._has_model_state.value and self.gt.applied_state.value:
            self.ct.calc_scale.disabled = False
            self.ct.calc_scale.tooltip = "Calculate camera scale relative to object."
        else:
            self.ct.calc_scale.disabled = True
            self.ct.calc_scale.tooltip="Apply General-Tab to activate."

    def _on_save_camerapath_btn_click(self, btn):
        if self.ct.fc.selected:
            parent = Path(self.ct.fc.selected).parent
            tmp_filename =Path(self.ct.fc.selected).stem
            filename = Path(tmp_filename).with_suffix(".json")

            out = parent.joinpath(filename)

            data = self.camerapaths[self.dc.datasets.value]
            data["coordinates"] = np.asarray(data["coordinates"]).tolist()

            with open(out, "w") as outfile:
                json.dump(data, outfile, indent = 4)

    def _on_load_camerapath_btn_click(self, btn):
        if Path(self.ct.fc.selected).suffix == ".json":
            with open(self.ct.fc.selected, "r") as json_file:
                    tmp = json.load(json_file)

            self.camerapaths[self.dc.datasets.value] = tmp
            self._update_cameraview_tab(change=self.dc.datasets.value)

    def _on_update_lightsetup_btn_click(self, btn):
        self.lightsetup.reload()
        self.lightconfigs[self.dc.datasets.value] = self.lightsetup.data

        if not self.config["datasets"][self.dc.datasets.value]["files"]["lightsetup"]:
            self.config["datasets"][self.dc.datasets.value]["files"]["lightsetup"] = f"configs/{self.dc.datasets.value}_lightsetup.json"

        # Update statusbar and check completeness of experiment
        self.datasetchecker.set_lighting_label_status(self.datasetchecker.is_lighting_applied(self.config["datasets"][self.dc.datasets.value]["files"]["lightsetup"]))
        self._experiment_complete_state.value = self._completeness_check(self.config)

        self.info_label.value = self._LBL_TEMPLATE.format(f"Applied current Blender lightsetup configuration to {self.dc.datasets.value} dataset.", 'green')

    def _on_lightsetup_visualize_change(self, change):
        if change.new:
            self.lightsetup.clear_scene()
            self.lightsetup.create()
        else:
            self.lightsetup.clear_scene()

    def _on_load_lightconfig_btn_click(self, btn):
        self.lightsetup.from_json(filepath=self.lt.fc.selected)
        self.lightconfigs[self.dc.datasets.value] = self.lightsetup.data

        if not self.config["datasets"][self.dc.datasets.value]["files"]["lightsetup"]:
            self.config["datasets"][self.dc.datasets.value]["files"]["lightsetup"] = f"configs/{self.dc.datasets.value}_lightsetup.json"

        if self.lt.visualize.value:
            self.lightsetup.clear_scene()
            self.lightsetup.create()

        self._experiment_complete_state.value = self._completeness_check(self.config)

    def _on_save_lightconfig_save_btn_click(self, btn):
        
        if self.lt.fc.selected:
            parent = Path(self.lt.fc.selected).parent
            tmp_filename =Path(self.lt.fc.selected).stem
            filename = Path(tmp_filename).with_suffix(".json")

            self.lightsetup.export_json(directory=parent, filename=filename)


    def _on_load_material_database_click(self, btn):
        materials = list_materials_of_blend_file(filepath=self.matt.fc.selected)

        for m in materials:
            if m in self.matt.material_model.options:
                materials.remove(m)

        self.matt.material_model.options=self.matt.material_model.options + tuple(materials)
        self.matt.material_base_plate.options=self.matt.material_base_plate.options + tuple(materials)

    def _on_use_baseplate_change(self, change):
        if change.new:
            self.matt.material_base_plate.disabled = False
        else:
            self.matt.material_base_plate.disabled = True

    def _on_apply_material_btn_click(self, btn):
        blend_file_materials = [x.name for x in bpy.data.materials]
        materials_to_load = set()

        if not self.matt.material_model.value in blend_file_materials:
            materials_to_load.add(self.matt.material_model.value)
            print(materials_to_load)
        
        if not self.matt.material_base_plate.disabled:
            if not self.matt.material_base_plate.value in blend_file_materials:
                materials_to_load.add(self.matt.material_base_plate.value)
                print(materials_to_load)
        
        print(materials_to_load)
        if materials_to_load:
            load_material(filepath=self.matt.fc.selected, materials=list(materials_to_load))

        self.config["datasets"][self.dc.datasets.value]["material"] = self.matt.material_model.value
        self.config["datasets"][self.dc.datasets.value]["base_plate_mat"] = self.matt.material_base_plate.value

        self.datasetchecker.set_material_label_status(self.datasetchecker.is_material_applied(self.config["datasets"][self.dc.datasets.value]["material"], self.config["datasets"][self.dc.datasets.value]["base_plate_mat"]))
        self._experiment_complete_state.value = self._completeness_check(self.config)

        self.info_label.value = self._LBL_TEMPLATE.format(f"Applied material configuration to {self.dc.datasets.value} dataset.", 'green')

    def _on_background_apply_btn_click(self, btn):
        if self.bt.fc.selected:
            self.background_images.update({self.dc.datasets.value: self.bt.fc.selected})

        if self.background_images[self.dc.datasets.value]:
            self.config["datasets"][self.dc.datasets.value]["files"]["background"] = f"configs/{Path(self.background_images[self.dc.datasets.value]).name}"

        # Update statusbar and check completeness of experiment
        self.datasetchecker.set_background_label_status(self.datasetchecker.is_background_applied(self.config["datasets"][self.dc.datasets.value]["files"]["background"]))
        self._experiment_complete_state.value = self._completeness_check(self.config)

        self.info_label.value = self._LBL_TEMPLATE.format(f"Applied background configuration to {self.dc.datasets.value} dataset.", 'green')

    def _on_save_experiment_btn_click(self, btn):
        try:
            self.info_label.value = self._LBL_TEMPLATE.format("Saving Experiment...", "orange")
            self._save_experiment()
            self.info_label.value = self._LBL_TEMPLATE.format(f"Experiment saved at {self.sr.fc.selected}.", "green")
        except Exception as e:
            self.info_label.value = self._LBL_TEMPLATE.format(f"Error: Saving failed with message: \"{e}\"", "red")


    def _on_run_experiment_btn_click(self, btn):
        self.sr.run_btn.disabled = True
        self.sr.save_btn.disabled = True 

        self.info_label.value = self._LBL_TEMPLATE.format("Creating Experiment... (this can take some time)", "orange")
        self._save_experiment()

        json_files = [x for x in Path(self.sr.fc.selected).glob("*.json")]
        
        # Find the right config file
        run_config =""
        for j in json_files:
            with open(j, "r") as json_file:
                cfg = json.load(json_file)
                try:
                    cfg["object_name"]
                    
                    run_config = str(j)
                    break
                except:
                    continue
        
        try:
            run.main(run_config)
            self.info_label.value = self._LBL_TEMPLATE.format(f"Experiment created at {self.sr.fc.selected}.", "green")
            self.sr.run_btn.disabled = False
            self.sr.save_btn.disabled = False 
        except Exception as e:
            self.info_label.value = self._LBL_TEMPLATE.format(f"Error: Run script failed with message: \"{e}\"", "red")
            self.sr.run_btn.disabled = False
            self.sr.save_btn.disabled = False 


    def _on_sr_label_change(self, change):
        if change.new!= "Select":
            self.datasetchecker.set_exp_directory_label_status(self.datasetchecker.is_exp_dir_set(self.sr.fc.selected))
            self._experiment_complete_state.value = self._completeness_check(self.config)

    def _save_experiment(self):
        save_dir = Path(self.sr.fc.selected)
        save_dir.mkdir(exist_ok=True)

        configs_dir = save_dir.joinpath("configs")
        configs_dir.mkdir(exist_ok=True)

        # Save blend file
        blend_filepath = save_dir.joinpath(self.sr.fc.selected_filename).with_suffix(".blend")
        save_blend_file(filepath=str(blend_filepath))

        # Copy background images
        for key in self.background_images:
            src = self.background_images[key]
            dst = configs_dir.joinpath(Path(src).name)

            if not dst.exists():
                shutil.copy2(src, dst)

        # Save lightconfigs
        for key in self.lightconfigs:
            self.lightsetup.from_dict(self.lightconfigs[key])
            self.lightsetup.export_json(directory=configs_dir, filename=str(Path(self.config["datasets"][key]["files"]["lightsetup"]).name))
              
        # Save Camerapaths
        for key in self.camerapaths:
            cp = CameraPath.from_dict(data=self.camerapaths[key])
            cp.export_json(directory=configs_dir, filename=str(Path(self.config["datasets"][key]["files"]["camera_config"]).name))


        # Save config
        config_path = save_dir.joinpath(f"{self.sr.fc.selected_filename}_config").with_suffix(".json")
        with open(config_path, "w") as outfile:
                json.dump(self.config, outfile, indent = 4)


    def _completeness_check(self, cfg:dict)->bool:
        is_full = True
        tmp_cfg = copy.deepcopy(cfg)

        # Set boolean values to true if false
        if not tmp_cfg["create_depth_normal"]: tmp_cfg["create_depth_normal"] = True
        if not tmp_cfg["use_base_plate"]: tmp_cfg["use_base_plate"] = True

        if not self.sr.fc.selected:
            return False

        is_full = all(tmp_cfg.values())
        if not is_full: return False

        for key in tmp_cfg["datasets"]:
            is_full = all(tmp_cfg["datasets"][key].values())
            if not is_full: return False

            is_full = all(tmp_cfg["datasets"][key]["files"].values())
            if not is_full: return False
            
        
        return True
        