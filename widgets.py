from pathlib import Path
from ipywidgets import *
from ipyfilechooser import FileChooser
import json
from blenderFBK import *
import io
import PIL.Image
import copy


class ExperimentLoader():
    def __init__(self):

        self.fc = FileChooser(value = None, show_only_dirs=True)
        self.load_btn = Button(
            description='Load Experiment',
            disabled= True,
            tooltip="Select experiment folder to load experiment."
            )

        self.fc_box = HBox([self.fc, self.load_btn])
        self.fc._select.observe(self.fc_button_change, "description")
        
        
    def instantiate(self):
        return HBox([self.fc_box])
        
            
    def fc_button_change(self, change):
        if self.fc._select != "Select":
            self.load_btn.disabled = False
        else:
            self.load_btn.disabled = True

        
    def _consistency_check(self) -> bool:        
        dir = Path(self.fc.selected)
        
        blend_files = [x for x in dir.glob("*.blend")]
        json_files = [x for x in dir.glob("*.json")]
   
        # Check existence of .blend file in experiment
        if len(blend_files) == 0:
            return False
        
        # Check if one json file is the config file
        is_config_file = False
        for j in json_files:
            with open(j, "r") as json_file:
                config = json.load(json_file)
                try:
                    config["object_name"]
                    is_config_file = True
                except:
                    continue
        
        if not is_config_file:
            return False
 
        # Check existence of configs folder in experiment
        if not dir.joinpath("configs").exists():
            return False
        
        return True
    

class DatasetChooser():
    def __init__(self):
        self.box = None
                
        self.datasets = Dropdown(
        options=[],
        description='Dataset:',
        disabled=False,
        style={'description_width': 'auto'}
        )
        self.datasets_box = HBox([self.datasets],layout=Layout(width='40%', display="flex", justify_content="flex-start"))

        self.textfield = Text(
        value='',
        placeholder='Type name of dataset to add',
        )
        
        self.add_btn = Button(
            description='Add Dataset',
            disabled= True,
        )
        
        self.del_btn = Button(
            description='Delete Dataset',
            disabled= True,
        )
        
        self.textfield.observe(self._add_button_change, "value")
        self.datasets.observe(self._del_button_change, "value")
        self._instantiate()
        
    def _add_button_change(self, change):
        if len(change.new) > 0:
            self.add_btn.disabled = False
        else:
            self.add_btn.disabled = True

    def _del_button_change(self, change):
        if change.new:
            self.del_btn.disabled = False
        else:
            self.del_btn.disabled = True
            
            
    def _instantiate(self):
        augmenters = HBox([self.textfield, self.add_btn, self.del_btn], laout=Layout(width='60%', display="flex", justify_content="flex-end"))
        self.box = HBox([self.datasets_box, augmenters], layout=Layout(width='100%', justify_content='space-between'))
        display(self.box)
            


class Options:
    def __init__(self):
        self.height = "390px"
        self.tab = Tab()
        
        self.general = Box()
        self.model = Box()
        self.material = Box()
        self.background = Box()
        self.cameraviews = Box()
        self.lighting = Box()
        
        self.general.layout.height = self.height
        self.model.layout.height = self.height
        self.material.layout.height = self.height
        self.background.layout.height = self.height
        self.cameraviews.layout.height = self.height
        self.lighting.layout.height = self.height
        
        self.children = [self._empty_dataset()] * 6
        
        self.tab.children = self.children
        self.tab.layout.height = self.height
        
        self.tab.set_title(0, "General")
        self.tab.set_title(1, "Model")
        self.tab.set_title(2, "Material")
        self.tab.set_title(3, "Background")
        self.tab.set_title(4, "Cameraviews")
        self.tab.set_title(5, "Lighting")
        
        display(self.tab)
        


    def _empty_dataset(self):
        return Label(value="Add a dataset. At least one dataset is needed to access options.")

class GeneralTab:
    def __init__(self) -> None:
        self.applied_state = Checkbox(value=False)

        self.layout = Layout(width='auto', display="flex", justify_content="flex-start")
        self.label_layout = Layout(width="130px", display="flex", justify_content="flex-end")
        self.style= {'description_width': '130px'}
        self.source = Text(description="Source:", placeholder="Enter source of the object", style=self.style, layout=self.layout)
        self.x_resolution = BoundedIntText(description="X:", min=1, max=5000, value=600, layout=Layout(width="150px"))
        self.y_resolution = BoundedIntText(description="Y:", min=1, max=5000, value=600, layout=Layout(width="150px"))
        self.cam_resolution = HBox([Label(value="Cam Resolution:", layout=self.label_layout), self.x_resolution, self.y_resolution], layout=self.layout)
        self.color_depth= BoundedIntText(description="Color Depth:", value=8, min=1, max=32, style=self.style)
        self.file_format = Dropdown(description="File Format:", value="PNG", options=["JPEG", "PNG", "OpenEXR", "TIFF", "BMP"], style=self.style, layout=self.layout)
        self.category1 = Text(description="Cat 1:", value="Blender")
        self.category2 = Text(description="Cat 2:", value="-")
        self.category3 = Text(description="Cat 3:", value="-")
        self.categories=HBox([Label(value="Categories:", layout=self.label_layout), self.category1, self.category2, self.category3], layout=self.layout)
        self.shaders_with_bg = Checkbox(value=True, description="Shaders with background")
        self.without_shaders = Checkbox(value=True, description="Without Shaders")
        self.shaders_without_bg = Checkbox(value=True, description="Shaders without background")
        self.attributes = HBox([Label(value="Attributes:", layout=self.label_layout), self.shaders_with_bg, self.shaders_without_bg, self.without_shaders], layout=self.layout)
        self.variant_num = Text(description="Variant Num:", value="base", style=self.style, layout=self.layout)
        self.create_depth_normal = Checkbox(value=True)
        self.create_d_n = HBox([Label(value="Create Depth Normal:"), self.create_depth_normal], layout=self.layout)
        self.device=Dropdown(options=["GPU", "CPU"], value="GPU", description="Device:", style=self.style, layout=self.layout)
        self.compute = Dropdown(options=["CUDA", "METAL","NONE", "OPTIX", "HIP"], value="CUDA", description="Computation:", style=self.style, layout=self.layout)
        self.compute_devices = HBox([self.device, self.compute], layout=self.layout)
        self.apply_general_btn = Button(description="Apply", disabled=True)
        self.apply_general_btn_box = Box([self.apply_general_btn], layout=Layout(width="auto", display="flex", justify_content="flex-end"))

        self.source.observe(self._is_filled, "value")
        self.x_resolution.observe(self._is_filled, "value")
        self.y_resolution.observe(self._is_filled, "value")
        self.color_depth.observe(self._is_filled, "value")
        self.file_format.observe(self._is_filled, "value")
        self.category1.observe(self._is_filled, "value")
        self.category2.observe(self._is_filled, "value")
        self.category3.observe(self._is_filled, "value")
        self.variant_num.observe(self._is_filled, "value")

    def _is_filled(self, _):
        fields = [self.source.value, self.x_resolution.value, self.y_resolution.value, self.color_depth.value, self.file_format.value, self.category1.value, self.category2.value, self.category3.value, self.variant_num.value]
        if all(fields):
            self.apply_general_btn.disabled = False
        else:
            self.apply_general_btn.disabled = True


    def _return_context(self):
        return VBox([self.source, self.cam_resolution, self.color_depth, self.file_format, self.categories, self.attributes, self.variant_num, self.create_d_n, self.compute_devices, self.apply_general_btn_box])

class ModelTab:
    def __init__(self):
        self._model_selected_state = Checkbox(value=False)

        self.layout = Layout(width='auto', display="flex", justify_content="flex-start")
        self.label_layout = Layout(width="130px", display="flex", justify_content="flex-end")
        self.style= {'description_width': '130px'}
        self.fc = FileChooser(filter_pattern=["*.stl", "*.obj"])
        self.upload_box = HBox([Label(value="Choose Model:", layout=self.label_layout), self.fc])
        self.object_name= Text(description="Object name:", placeholder="Enter name of the object", style=self.style, layout=self.layout)
        self.labelID = Text(description="Label-ID:", placeholder="Enter label-ID of the object", style=self.style, layout=self.layout)
        self.baseplate = Checkbox(value=True, layout=self.layout)
        self.baseplate_box = HBox([Label(value="Use base plate:", layout=self.label_layout), self.baseplate])

        self.apply_model_btn = Button(description="Apply", disabled=True)
        self.apply_model_btn_box = Box([self.apply_model_btn], layout=Layout(width="auto", display="flex", justify_content="flex-end"))

        self.fc._select.observe(self._model_selected, "description")

    def _model_selected(self, change):
        if self.fc._select != "Select":
            self._model_selected_state.value = True
        else:
            self._model_selected_state.value = False


    def _return_context(self):
        return VBox([self.upload_box, self.object_name, self.labelID, self.baseplate_box, self.apply_model_btn_box])


class MaterialTab:
    def __init__(self):
        self.layout = Layout(width='auto', display="flex", justify_content="flex-start")
        self.style= {'description_width': '130px'}

        self.init_mats = [x.name for x in bpy.data.materials if x.name != "Dots Stroke"]

        self.fc  = FileChooser()
        self.load_database_btn = Button(description="Load Database", disabled=True)

        self.fc_box = HBox([self.fc, self.load_database_btn], layout=self.layout)

        self.material_model = Dropdown(options=self.init_mats, description="Model Material:", style=self.style)
        self.material_base_plate = Dropdown(options=self.init_mats, description="Base_plate Material:", style=self.style)

        self.apply_btn = Button(description="Apply")

        self.apply_btn_box = HBox([self.apply_btn], layout= Layout(width="auto", display="flex", justify_content="flex-end"))

        self.fc._select.observe(self._on_fc_change, "description")

    def _on_fc_change(self, change):
        if change.new != "Select":
            self.load_database_btn.disabled = False
        else:
            self.load_database_btn.disabled = True

    def return_context(self):
        return VBox([self.fc_box, self.material_model, self.material_base_plate, self.apply_btn_box], layout=self.layout)


class BackgroundTab:
    def __init__(self) -> None:
        self.layout = Layout(width='auto', display="flex", justify_content="flex-start")
        self.preview_height = 150

        self.fc = FileChooser(filter_pattern=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"], layout=self.layout)
        self.background_preview = Image(layout=Layout(width="auto", height=f"{self.preview_height}px", object_fit="contain"))
        self.apply_btn = Button(description="Apply")
        self.apply_btn_box = Box([self.apply_btn], layout=Layout(width="auto", display="flex", justify_content="flex-end"))

        self.fc._label.observe(self._on_label_change, "value")

    def _on_label_change(self, change):
        try:
            with PIL.Image.open(self.fc.selected) as image:
                img_byte_arr = io.BytesIO()
                height_percent = (self.preview_height*2 / float(image.size[1]))
                width_size = int((float(image.size[0]) * float(height_percent)))
                scaled_image = image.resize((width_size, self.preview_height*2), PIL.Image.NEAREST)  
                scaled_image.save(img_byte_arr, format=image.format)
                self.background_preview.value = img_byte_arr.getvalue()
        except:
            pass

    def return_context(self):
        return VBox([self.fc, self.background_preview, self.apply_btn_box])


class CameraviewTab:
    def __init__(self):
        self.layout = Layout(width='auto', display="flex", justify_content="flex-start")
        self.left_column_layout = Layout(width="70%", display="flex", justify_content="flex-start")
        self.right_column_layout = Layout(width="auto", display="flex", justify_content="flex-start")
        self.label_layout = Layout(width="130px", display="flex", justify_content="flex-end")
        self.style= {'description_width': '130px'}
        
        # Elements and logic for custom camerapaths
        self.cc = CustomCamerapath()

        self.fc = FileChooser(value=None)
        self.load_btn = Button(description="Load", style=self.style, disabled=True)
        self.fc._select.observe(self.fc_load_btn_change, "description")
        self.fc._select.observe(self.fc_save_btn_change, "description")

        self.fc_box = HBox([self.fc, self.load_btn], layout=self.layout)

        self.method = Dropdown(options=["Spherical Coordinates", "Icosphere", "Custom"], value="Spherical Coordinates", description="Method:", style=self.style, layout=self.layout)
        self.canvas = Output(layout=Layout(height="250px"))

        self.scale = FloatSlider(value = 1, min=0.0, max=50.0, style=self.style, layout=Layout(width="90%"), description="Scale:")
        self.subdiv = IntSlider(min=0, value =1, max=5, style=self.style, layout=self.layout, description="Subdivisions:")

        self.sph_degree_range = FloatRangeSlider(min=-180.0, max=180.0, value=[0, 90], description="Range:", readout_format='.1f', style=self.style, layout=self.layout)
        self.sph_rotations = IntRangeSlider(min=1.0, max=50.0, value=[10, 30], description="Range of Rotations:", style=self.style, layout=self.layout)
        self.sph_offset = FloatSlider(value=0, min=0, max = 2, description="Z-Offset:", style=self.style, layout=self.layout)


        self.calculate_btn = Button(description="Apply")
        self.save_btn = Button(description="Save as json", disabled=True)

        self.calculate_btn_box = Box([self.calculate_btn, self.save_btn], layout=Layout(width="auto", display="flex", justify_content="flex-end"))
        

        self.num_layers = IntSlider(value=1, min=1, max=20, description="Number of layers:", style=self.style, layout=self.layout)

        self.calc_scale = Checkbox(value=False, disabled=True, description="Relative to object", description_tooltip="Apply General-Tab to activate.")
        self.scale_box = HBox([self.scale, self.calc_scale], layout=self.layout)
        self.rotation = FloatSlider(min=0.0, max=360.0, style=self.style, layout=self.layout, description="Z-Rotation:")
        self.ico_options = Dropdown(options=["upper_half", "lower_half", "full_sphere"], style=self.style, layout=self.layout, description="Camera positioning:")

        self.left_column = VBox([self.fc_box, self.method, self.canvas], layout=self.left_column_layout)

        self.ico_args = VBox([self.scale_box, self.subdiv, self.rotation, self.ico_options, self.calculate_btn_box])
        self.sph_args = VBox([self.sph_degree_range, self.num_layers, self.sph_rotations, self.scale_box, self.sph_offset, self.calculate_btn_box])

        self.custom_args = VBox([self.cc.return_context(), self.calculate_btn_box])

        self.visualize = Checkbox(value=False, description="Visualize in Blender", disabled=True)
        self.vis_slider = IntSlider(value=0, min=0, disabled=True)

        self.right_column = VBox([self.visualize, self.vis_slider], layout= self.right_column_layout)

        self._instantiate()
        self.method.observe(self._on_method_change, "value")

    def _instantiate(self, method="Spherical Coordinates"):
        self.canvas.clear_output()

        if method == "Spherical Coordinates":
            with self.canvas:
                display(self.sph_args)
        elif method == "Icosphere":
            with self.canvas:
                display(self.ico_args)
        else:
            with self.canvas:
                display(self.custom_args)

    def _on_method_change(self, change):
        self._instantiate(method=change.new)
    
    def _return_context(self):
        return HBox([self.left_column, self.right_column])
    
    def fc_load_btn_change(self, change):
        if self.fc._select != "Select":
            self.load_btn.disabled = False
        else:
            self.load_btn.disabled = True

    def fc_save_btn_change(self, change):
        if self.fc._select != "Select":
            self.save_btn.disabled = False
        else:
            self.save_btn.disabled = True


class LightingTab:
    def __init__(self) -> None:
        self.layout = Layout(width='auto', display="flex", justify_content="flex-start")

        self.info_text = Label(value= "Note: Lightsetup needs to be created within blender!")
        self.fc = FileChooser()
        self.load_btn = Button(description= "Load", disabled= True)
        self.visualize = Checkbox(value=False, description="Visualize in Blender")

        self.update_btn = Button(description= "Update current setup")
        self.save_btn = Button(description= "Save as json", disabled= True)

        self.fc_box = HBox([self.fc, self.load_btn, self.visualize], layout= self.layout)
        self.btn_box = HBox([self.update_btn, self.save_btn], layout= Layout(width="auto", display="flex", justify_content="flex-end"))

        self.fc._select.observe(self._on_fc_change, "description")

    def _on_fc_change(self, change):
        if change.new != "Select":
            self.load_btn.disabled = False
            self.save_btn.disabled = False
        else:
            self.load_btn.disabled = True
            self.save_btn.disabled = True
    
    def return_context(self):
        return VBox([self.info_text, self.fc_box, self.btn_box], layout=self.layout)


class Save_Run_Experiment:
    def __init__(self) -> None:
        self.layout = Layout(width="auto", display="flex", justify_content="flex-end")

        self.fc = FileChooser()
        self.fc._filename.placeholder = "experiment name"
        self.save_btn = Button(description="Save Experiment", disabled=True)
        self.run_btn = Button(description="Create Experiment", disabled=True)

        self.box = HBox([self.fc, self.save_btn, self.run_btn], layout= self.layout)

        self._initialize()

    def _initialize(self):
        display(self.box)


class CustomCoordinatesLayout():
    """ 
    Visualization of the custom Camerapath coordinate selector and logic.
    """
    def __init__(self):
        # Lists for all coordinates, using a select widget for x coords to use the stateful properties 
        self.all_x_coords= []
        self.all_y_coords= []
        self.all_z_coords = []
        
        self.rows = 10
        self.current_upper_index = 0
        self.current_lower_index = self.rows
        self.box_layout= Layout(widht="100%", display="flex", align_items="stretch", flex_flow="row")
        self.list_layout= Layout(widht="auto", flex="1 auto")
        self.button_layout= Layout(width="35px", height="35px")
        self.xcoord = Select(rows=self.rows, layout=self.list_layout, options=[])
        self.ycoord = Select(rows=self.rows, layout=self.list_layout, options=[])
        self.zcoord = Select(rows=self.rows, layout=self.list_layout, options=[])
        self.up_button= Button(layout=self.button_layout, icon="arrow-up", disabled=True)
        self.down_button = Button(layout=self.button_layout, icon="arrow-down", disabled=True)

        self.controls = VBox([self.up_button, self.down_button])

   
        self.xcoord.observe(self._on_list_index_change, "index")
        self.ycoord.observe(self._on_list_index_change, "index")
        self.zcoord.observe(self._on_list_index_change, "index")

        self.up_button.on_click(self._up_btn_click)
        self.down_button.on_click(self._down_btn_click)

        self.instantiate()


    def return_context(self):
        return HBox([self.xcoord, self.ycoord, self.zcoord, self.controls], layout=self.box_layout)


    def _up_btn_click(self, btn):
        # If Index is not on top of visible coordiantes, go up
        if self.xcoord.index:
            self.xcoord.index = self.xcoord.index -1
        # If it is on top, check if there are more coordinates above and scroll up
        else:
            if self.current_upper_index > 0:
                self.current_upper_index = self.current_upper_index - 1
                self.current_lower_index = self.current_lower_index - 1 
                   

                tmp_x = self.all_x_coords[self.current_upper_index:self.current_lower_index]
                tmp_y = self.all_y_coords[self.current_upper_index:self.current_lower_index]
                tmp_z = self.all_z_coords[self.current_upper_index:self.current_lower_index]

                self.xcoord.options = tmp_x
                self.ycoord.options = tmp_y
                self.zcoord.options = tmp_z 

                self.xcoord.index = 1 
                self.xcoord.index = 0


    def _down_btn_click(self, btn):
        self.up_button.disabled = False

        if self.xcoord.index < len(self.xcoord.options)-1:
            self.xcoord.index = self.xcoord.index +1
        else:
            if len(self.all_z_coords) > self.current_lower_index:
                self.current_upper_index = self.current_upper_index + 1
                self.current_lower_index = self.current_lower_index + 1

                tmp_x = self.all_x_coords[self.current_upper_index:self.current_lower_index]
                tmp_y = self.all_y_coords[self.current_upper_index:self.current_lower_index]
                tmp_z = self.all_z_coords[self.current_upper_index:self.current_lower_index]

                self.xcoord.options = tmp_x
                self.ycoord.options = tmp_y
                self.zcoord.options = tmp_z 
                
                self.xcoord.index = self.rows-1
               

    def _on_list_index_change(self,change):
        # Try to set all indices to the same value
        try:
            self.xcoord.index = change.new
            self.ycoord.index = change.new
            self.zcoord.index = change.new
        except:
            pass

        # Enable and Disable Button relative to position in list
        if len(self.all_x_coords) > 1:
            if self.xcoord.index == 0 and self.current_upper_index == 0:
                self.up_button.disabled = True
            else:
                self.up_button.disabled = False
            
            if self.xcoord.index == len(self.xcoord.options)-1 and self.current_lower_index == len(self.all_x_coords):
                self.down_button.disabled = True
            else:
                self.down_button.disabled = False
        else:
            self.up_button.disabled = True
            self.down_button.disabled = True


    # Fills the select boxes with data at the start of the programme or an experiment load
    def instantiate(self):
        # If there are coordinates
        if self.all_x_coords:
            self.current_upper_index = 0
            # If number of coordiantes is less than visual rows, update all options
            if len(self.all_x_coords) <= self.rows:
                self.xcoord.options = self.all_x_coords
                self.ycoord.options = self.all_y_coords
                self.zcoord.options = self.all_z_coords
                # Update indices
                self.current_lower_index = len(self.all_x_coords)
                self.xcoord.index = 0
                self.xcoord.index = 0
            else:
                # show the n=rows first coordinates
                self.current_lower_index = self.rows
                tmp_x = self.all_x_coords[self.current_upper_index:self.current_lower_index]
                tmp_y = self.all_y_coords[self.current_upper_index:self.current_lower_index]
                tmp_z = self.all_z_coords[self.current_upper_index:self.current_lower_index]

                self.xcoord.options = tmp_x
                self.ycoord.options = tmp_y
                self.zcoord.options = tmp_z 

                self.xcoord.index = 0
                self.xcoord.index = 0


class CustomCamerapath():
    def __init__(self) -> None:
        self.cw = CustomCoordinatesLayout()
        self.box_layout = Layout(widht="100%", display="flex", align_items="stretch", flex_flow="row")
        self.textbox_layout= Layout(widht="auto", flex="1 auto")
        self.x = FloatText(layout=self.textbox_layout, description="X:", style={"description_width": "auto"})
        self.y = FloatText(layout=self.textbox_layout, description="Y:", style={"description_width": "auto"})
        self.z = FloatText(layout=self.textbox_layout, description="Z:", style={"description_width": "auto"})

        self.add_btn = Button(description="Add", layout=Layout(width="80px"))
        self.del_button = Button(description="Delete", layout=Layout(width="80px"), disabled=True)

        self.add_btn.on_click(self._add_btn_click)
        self.del_button.on_click(self._del_button_click)

    def return_context(self):
        input_controls = HBox([self.x, self.y, self.z, self.add_btn, self.del_button], layout=self.box_layout)
        return VBox([input_controls, self.cw.return_context()])

    def _add_btn_click(self, btn):
        # If there are less coordinates than visible rows just update all options
        if len(self.cw.all_x_coords) < self.cw.rows:
            self.cw.all_x_coords.append(self.x.value)
            self.cw.all_y_coords.append(self.y.value)
            self.cw.all_z_coords.append(self.z.value)

            self.cw.instantiate()
            self.cw.xcoord.index = len(self.cw.xcoord.options)-1
        # else add new coordiante to lists and show last n=rows entries
        else:
            self.cw.all_x_coords.append(self.x.value)
            self.cw.all_y_coords.append(self.y.value)
            self.cw.all_z_coords.append(self.z.value)

            self.cw.current_lower_index = len(self.cw.all_x_coords)
            self.cw.current_upper_index = self.cw.current_lower_index - self.cw.rows

            temp_x = self.cw.all_x_coords[self.cw.current_upper_index:self.cw.current_lower_index]
            temp_y = self.cw.all_y_coords[self.cw.current_upper_index:self.cw.current_lower_index]
            temp_z = self.cw.all_z_coords[self.cw.current_upper_index:self.cw.current_lower_index]

            self.cw.xcoord.options = temp_x
            self.cw.ycoord.options = temp_y
            self.cw.zcoord.options = temp_z

            self.cw.xcoord.index = len(self.cw.xcoord.options)-1
        
        # Reset input fields
        self.x.value = self.y.value = self.z.value = 0.0

        # activate delete buttons as there is at least one option after add button click
        self.del_button.disabled = False


    def _del_button_click(self, btn):
        current_index = self.cw.xcoord.index
        if len(self.cw.all_x_coords) <= self.cw.rows:

            self.cw.all_x_coords.pop(current_index)
            self.cw.all_y_coords.pop(current_index)
            self.cw.all_z_coords.pop(current_index) 

            self.cw.xcoord.options = self.cw.all_x_coords
            self.cw.ycoord.options = self.cw.all_y_coords
            self.cw.zcoord.options = self.cw.all_z_coords

            # Update indices
            self.cw.current_lower_index = len(self.cw.all_x_coords)
            if current_index < len(self.cw.xcoord.options) and current_index > 0:
                self.cw.xcoord.index = current_index
            elif current_index ==  len(self.cw.xcoord.options) and current_index > 0:
                self.cw.xcoord.index = len(self.cw.xcoord.options)-1
                self.cw.xcoord.index = len(self.cw.xcoord.options)-1
            else:
                # ignore index at empty list
                pass

        else:
            tmp_index = self.cw.current_upper_index + current_index
            if self.cw.current_lower_index == len(self.cw.all_x_coords):
                self.cw.all_x_coords.pop(tmp_index)
                self.cw.all_y_coords.pop(tmp_index)
                self.cw.all_z_coords.pop(tmp_index)  

                self.cw.current_lower_index = self.cw.current_lower_index - 1
                self.cw.current_upper_index = self.cw.current_upper_index - 1

                temp_x = self.cw.all_x_coords[self.cw.current_upper_index:self.cw.current_lower_index]
                temp_y = self.cw.all_y_coords[self.cw.current_upper_index:self.cw.current_lower_index]
                temp_z = self.cw.all_z_coords[self.cw.current_upper_index:self.cw.current_lower_index]

                self.cw.xcoord.options = temp_x
                self.cw.ycoord.options = temp_y
                self.cw.zcoord.options = temp_z

                self.cw.xcoord.index = len(self.cw.xcoord.options)-1

            else: 

                self.cw.all_x_coords.pop(tmp_index)
                self.cw.all_y_coords.pop(tmp_index)
                self.cw.all_z_coords.pop(tmp_index) 

                temp_x = self.cw.all_x_coords[self.cw.current_upper_index:self.cw.current_lower_index]
                temp_y = self.cw.all_y_coords[self.cw.current_upper_index:self.cw.current_lower_index]
                temp_z = self.cw.all_z_coords[self.cw.current_upper_index:self.cw.current_lower_index]

                self.cw.xcoord.options = temp_x
                self.cw.ycoord.options = temp_y
                self.cw.zcoord.options = temp_z

                self.cw.xcoord.index = current_index


        if len(self.cw.all_x_coords) == 0:
            self.del_button.disabled = True

    
class DatasetChecker:

    """ 
    Visual statusbar that indicates the applied configs and methods to check the completeness of the configs.
    """

    def __init__(self) -> None:
        self._LBL_TEMPLATE = '<span style="color:{1};">{0}</span>'
        self.box_layout = Layout(widht="100%", display="flex", align_items="stretch", flex_flow="row")
        self.label_layout = Layout(widht="auto", flex="1 1 0%")

        self.description = Label(value="Dataset status:")
        self.general_label = HTML(value=self._LBL_TEMPLATE.format("General", 'black'), layout=self.label_layout)
        self.model_label = HTML(value=self._LBL_TEMPLATE.format("Model", 'black'), layout=self.label_layout)
        self.material_label = HTML(value=self._LBL_TEMPLATE.format("Material", 'black'), layout=self.label_layout)
        self.background_label = HTML(value=self._LBL_TEMPLATE.format("Background", 'black'), layout=self.label_layout)
        self.cameraviews_label = HTML(value=self._LBL_TEMPLATE.format("Cameraviews", 'black'), layout=self.label_layout)
        self.lighting_label = HTML(value=self._LBL_TEMPLATE.format("Lighting", 'black'), layout=self.label_layout)
        self.exp_dir_label = HTML(value=self._LBL_TEMPLATE.format("Experiment Directory", 'black'), layout=self.label_layout)

        self._instantiate()

    def _instantiate(self):
        display(HBox([self.description, self.general_label, self.model_label, self.material_label, self.background_label, self.cameraviews_label, self.lighting_label, self.exp_dir_label], layput=self.box_layout))

    def set_general_label_status(self, is_appled:bool):
        if is_appled:
            self.general_label.value = self._LBL_TEMPLATE.format("General ✓", 'green')
        else:
            self.general_label.value = self._LBL_TEMPLATE.format("General", 'black')

    def set_model_label_status(self, is_appled:bool):
        if is_appled:
            self.model_label.value = self._LBL_TEMPLATE.format("Model ✓", 'green')
        else:
            self.model_label.value = self._LBL_TEMPLATE.format("Model", 'black')

    def set_material_label_status(self, is_appled:bool):
        if is_appled:
            self.material_label.value = self._LBL_TEMPLATE.format("Material ✓", 'green')
        else:
            self.material_label.value = self._LBL_TEMPLATE.format("Material", 'black')
    
    def set_background_label_status(self, is_appled:bool):
        if is_appled:
            self.background_label.value = self._LBL_TEMPLATE.format("Background ✓", 'green')
        else:
            self.background_label.value = self._LBL_TEMPLATE.format("Background", 'black')

    def set_cameraviews_label_status(self, is_appled:bool):
        if is_appled:
            self.cameraviews_label.value = self._LBL_TEMPLATE.format("Cameraviews ✓", 'green')
        else:
            self.cameraviews_label.value = self._LBL_TEMPLATE.format("Cameraviews", 'black')

    def set_lighting_label_status(self, is_appled:bool):
        if is_appled:
            self.lighting_label.value = self._LBL_TEMPLATE.format("Lighting ✓", 'green')
        else:
            self.lighting_label.value = self._LBL_TEMPLATE.format("Lighting", 'black')

    def set_exp_directory_label_status(self, is_appled:bool):
        if is_appled:
            self.exp_dir_label.value = self._LBL_TEMPLATE.format("Experiment Directory ✓", 'green')
        else:
            self.exp_dir_label.value = self._LBL_TEMPLATE.format("Experiment Directory", 'black')

    def update_statusbar(self, general_applied, model_applied, material_applied, bg_applied, cameraviews_applied, lighting_applied, exp_dir_applied):
        self.set_general_label_status(general_applied)
        self.set_model_label_status(model_applied)
        self.set_material_label_status(material_applied)
        self.set_background_label_status(bg_applied)
        self.set_cameraviews_label_status(cameraviews_applied)
        self.set_lighting_label_status(lighting_applied)
        self.set_exp_directory_label_status(exp_dir_applied)

    #-------------------------------------------------------------------
    # Consistency Checks
    #-------------------------------------------------------------------
    def is_generel_applied(self, config:dict) -> bool:
        tmp_config = copy.deepcopy(config)
        
        # Remove all config entries that aren't related to the general tab
        tmp_config.pop("object_name")
        tmp_config.pop("labelID")
        tmp_config.pop("datasets")

        # Set false values to true to use all()
        if not tmp_config["create_depth_normal"]: tmp_config["create_depth_normal"] = True
        if not tmp_config["use_base_plate"]: tmp_config["use_base_plate"] = True

        is_complete = all(tmp_config.values())
        del tmp_config

        return is_complete

    def is_model_applied(self, model_name:str, label_id:str)->bool:
        if model_name and label_id:
            return True
        else:
            return False

    def is_material_applied(self, object_mat:str, base_plate_mat:str)->bool:
        if object_mat and base_plate_mat:
            return True
        else:
            return False

    def is_background_applied(self, background:bool)->bool:
        if background:
            return True
        else:
            return False

    def is_cameraviews_applied(self, camerapath:str)->bool:
        if camerapath:
            return True
        else:
            return False

    def is_lighting_applied(self, lightconfig:str)->bool:
        if lightconfig:
            return True
        else:
            return False

    def is_exp_dir_set(self, fc_selected:str):
        if fc_selected:
            return True
        else:
            return False