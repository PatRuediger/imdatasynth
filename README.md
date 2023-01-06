# ImDataSynth
A GUI to control a blender instance wrapped inside a jupyter-notebook kernel. 

The required data files and tutrial movies can be downloaded here: https://seafile.rlp.net/d/a582fd75539f44998df7/

## Setup
1. Install python 3.10 from https://www.python.org 
2. Download and install blender from https://www.blender.org/download/ 
- **Note: (At least) Blender 3.2 is required**
3. Follow the instruction on https://pypi.org/project/blender-notebook/ to wrap blender inside a jupyter kernel
4. Install the following packages, e.g. using pip:
```
pip install ipywidgets
pip install ipyfilechooser
pip install Pillow
pip install pandas
```
If using Windows:
```
pip install Windows-curses
```
5. Set the kernel to "blender" in gui.ipynb
- Note: Try restarting your system if the blender kernel is missing
6. Run gui.ipynb


## Notes
- GUI can still be buggy, if you find any bug add it to the "Known Bugs" list
- Custom Camerapaths is still under development 


## TO-DO
- **Refactoring, code is a mess at the moment** 
- Implement custom camerapaths
- Create environment.yaml (First implement custom camerapaths, as further packages are needed for this)


## Known Bugs
- Add bug if you find any

## Special Thanks
A special thanks for the gui development to my colleague Edwin.
