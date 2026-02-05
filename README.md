# DeepNCR： Native Contact Ratio as a Receptor-Aware Topological Metric for Molecular Docking

![fig1-0204](https://github.com/user-attachments/assets/5ac2c293-6bf1-4e99-8506-819e70734c75)
This repository provides a complete pipeline for **pose optimization**, **docking score prediction**, and **model retraining** using DeepNCR.

## 1. Pose Optimization Using Our Method

To perform pose optimization with our method, run the following command from the **root directory**:

```bash
bash run_pose_optimization.sh inputs.dat
````

* `inputs.dat` specifies the protein–ligand pairs to be optimized.
* The optimized poses will be generated according to the configuration defined in the script.


## 2. Scoring Optimized Poses with the Trained Model

To score docking poses using our trained model:

1. Edit the following paths in the script:

   * `protein_folder`
   * `decoy_folder`
   * `output_folder`

2. Run the scoring script directly:

```bash
python scoring/scripts/docking_score_quick.py
```

The output will contain the predicted docking scores for each pose.


## 3. Retraining DeepNCR

To retrain the DeepNCR model, feature generation and model training are required.

### 3.1 Feature Generation

Navigate to the `retrain` directory and generate features separately:

```bash
python generate_features_RMSD.py -inp inputs.dat -out data_rmsd.pkl
python generate_features_contact_num_ratio_12.py -inp inputs.dat -out data_contact.pkl
```

* `inputs.dat` defines the training samples.
* Each script generates a feature file in `.pkl` format.

### 3.2 Merge Feature Files

Merge the generated feature files into a single dataset:

```bash
python merge_pkl.py
```

### 3.3 Model Training

Finally, modify the data paths if necessary and run the training script:

```bash
python train.py
```

This will retrain the DeepNCR model using the newly generated features.


## 🛠 Requirements

Ensure you have the following dependencies installed:

* torch>=2.0.0
* torch-geometric==2.6.1
* biopython==1.85
* rdkit==2024.3.4
* vina==1.2.7
* numpy==1.26.4
* pandas==2.2.3
* scipy==1.13.1
* scikit-learn==1.5.2
* joblib==1.4.2
* tqdm==4.67.1


## Notes

* Please ensure all paths are correctly configured before running each step.
* Python version and required dependencies should be installed as specified in the environment setup.
