# Transformer-Graph-Network-for-Coronary-plaque-localization-in-CCTA
Transformer Graph Network for Coronary plaque localization in CCTA

# Format your data
create the folder data/patients/ctscan, data/patients/coronarytree
* ctscan are isotropic ctscan volumes (3d numpy array)
* coronarytree are coordinates points (2d numpy array Nx3), annotations for each point (2d numpy array Nx1) and an adjacency matrix (2d numpy array NxN)

# Run the training
python TrainCNNATT.py --train_data_path=<folder with train data> --valdi_data_path=<folder with valid data>

# Run the test
python TestCANNATT.py --test_data_path=<folder with test data> --ckpt=<log path to checkpointfile.ckpt>
