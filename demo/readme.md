# Whale Verification/Identification System

A graphical interface-based system for whale verification and identification. Features include drag-and-drop image input, whale identification and verification capabilities.
## Installation

Ensure Python 3.8 is installed.
Other necessary dependencies are:
-tensorflow
-h5py
-Pillow
-numpy
-scikit-learn
-torch
-torchvision
-matplotlib
-tqdm
-tkinterdnd2
-opencv



Which can be conveniently installed through:
pip install -r requirements.txt

## Usage

1. Run: `python main_interface.py` Preferably from Visual Studio Code

2. In the graphical interface, choose between:
   - Verification: Compare two whale images. Return whether the two whales belong to the same specimen. 
   - Identification: Identify the id of a given whale image, between the enrolled ones.

3. For identification, allow time (about 30 seconds and then the interface will open) for the model to load before proceeding.

4. Drag and drop image(s) onto the interface for analysis. 

5. Preprocess image before both verification and identification

6. Start verification/identification process



## Demo Images

For convenience, example images are provided in:
- `demo/Images/identification/`
- `demo/Images/verification/` 