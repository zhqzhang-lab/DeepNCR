# Native Contact Ratio as a Receptor-Aware Topological Metric for Molecular Docking

好嘞，我帮你直接整理成**可直接粘到 `README.md` 里的 Markdown**，结构清晰、步骤标准，偏科研/代码仓库风格 👍

---

````markdown
# DeepNCR Pipeline

This repository provides a complete pipeline for **pose optimization**, **docking score prediction**, and **model retraining** using DeepNCR.

---

## 1. Pose Optimization Using Our Method

To perform pose optimization with our method, run the following command from the **root directory**:

```bash
bash run_pose_optimization.sh inputs.dat
````

* `inputs.dat` specifies the protein–ligand pairs to be optimized.
* The optimized poses will be generated according to the configuration defined in the script.

---

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

---

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

---

## Notes

* Please ensure all paths are correctly configured before running each step.
* Python version and required dependencies should be installed as specified in the environment setup.

```

---

如果你愿意，下一步我也可以帮你加上：

- 📦 **Environment / Dependencies**
- 📁 **Directory Structure**
- 📊 **Example Output**
- 📖 **Citation / Reference**

直接说你打算投会议还是期刊，我可以按标准帮你再润一版 README。
```
