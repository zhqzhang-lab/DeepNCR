import os
import sys
import pandas as pd

# ==========================================
# Configuration
# ==========================================
MOL2_PKL_PATH = "pkl_mol2/inp_refine_50000.pkl"
CONTACT_PKL_PATH = "pkl_contact/inp_refine_50000_pdbqt.pkl"
OUTPUT_DIR = "pkl_merge"
OUTPUT_FILENAME = "inp_refine_50000.pkl"

# Columns used for duplicate detection to ensure data integrity
DUPLICATE_CHECK_COLS = [
    "index_col", 
    "r6_GLY-C_C", "r1_GLY-C_C", 
    "r6_GLY-C_O", "r1_GLY-C_O",
    "r6_GLY-C_N", "r1_GLY-C_N",
    "r6_MET-N_O", "r1_MET-N_O",
    "r6_CYS-N_O", "r1_CYS-N_O"
]

def merge_datasets():
    # 1. Load Data
    if not os.path.exists(MOL2_PKL_PATH) or not os.path.exists(CONTACT_PKL_PATH):
        print(f"Error: Input files not found.\nCheck {MOL2_PKL_PATH} and {CONTACT_PKL_PATH}")
        sys.exit(1)

    print("Loading datasets...")
    df_mol2 = pd.read_pickle(MOL2_PKL_PATH)
    df_contact = pd.read_pickle(CONTACT_PKL_PATH)
    
    print(f"Initial MOL2 count: {len(df_mol2)}")
    print(f"Initial CONTACT count: {len(df_contact)}")

    # 2. Reset Index for Merging
    # Ensure 'index_col' doesn't already exist to avoid conflicts
    if "index_col" in df_mol2.columns or "index_col" in df_contact.columns:
        raise ValueError("Input data already contains 'index_col'. Please check source files.")

    df_mol2 = df_mol2.reset_index().rename(columns={"index": "index_col"})
    df_contact = df_contact.reset_index().rename(columns={"index": "index_col"})

    # 3. Preprocessing to Handle Float Precision Issues
    # Round specific feature columns to ensure consistent matching
    # Note: df.columns[1] was originally used. Ensure "r6_GLY-C_C" corresponds to the correct feature.
    if "r6_GLY-C_C" not in df_mol2.columns:
         # Fallback if column name isn't found (based on original iloc[:, 1])
         # But safer to assume column names exist in production pkls
         pass 

    df_mol2["r6_GLY-C_C"] = df_mol2["r6_GLY-C_C"].round(3)
    df_contact["r6_GLY-C_C"] = df_contact["r6_GLY-C_C"].round(3)

    # 4. Remove Duplicates
    print("Checking for duplicates...")
    
    # Check if all required columns exist before filtering
    missing_cols = [c for c in DUPLICATE_CHECK_COLS if c not in df_mol2.columns]
    if missing_cols:
        print(f"Warning: Columns {missing_cols} missing in df_mol2, skipping full duplicate check.")
    else:
        print(f"MOL2 Duplicates: {df_mol2[DUPLICATE_CHECK_COLS].duplicated().sum()}")
        df_mol2 = df_mol2.drop_duplicates(subset=DUPLICATE_CHECK_COLS)

    missing_cols_contact = [c for c in DUPLICATE_CHECK_COLS if c not in df_contact.columns]
    if not missing_cols_contact:
        print(f"CONTACT Duplicates: {df_contact[DUPLICATE_CHECK_COLS].duplicated().sum()}")
        df_contact = df_contact.drop_duplicates(subset=DUPLICATE_CHECK_COLS)

    # Verify key uniqueness
    print(f"MOL2 Key Duplicates: {df_mol2[['index_col', 'r6_GLY-C_C']].duplicated().sum()}")
    print(f"CONTACT Key Duplicates: {df_contact[['index_col', 'r6_GLY-C_C']].duplicated().sum()}")

    # 5. Prepare for Merge
    # Assume the last column of df_mol2 is the label/target (e.g., 'rmsd')
    rmsd_col_name = df_mol2.columns[-1]
    
    # Keep only linking keys and the target variable from df_mol2
    df_mol2_subset = df_mol2[["index_col", "r6_GLY-C_C", rmsd_col_name]].copy()

    # 6. Merge DataFrames
    # Inner join on index name and the specific feature value
    print("Merging datasets...")
    merged_df = df_contact.merge(
        df_mol2_subset,
        on=["index_col", "r6_GLY-C_C"],
        how="inner",
        suffixes=("", "_target")
    )

    # 7. Post-processing
    # Restore original index
    merged_df = merged_df.set_index("index_col")

    # Rename the RMSD column if it got suffixed, or identify it
    if f"{rmsd_col_name}_target" in merged_df.columns:
        target_col = f"{rmsd_col_name}_target"
    else:
        target_col = rmsd_col_name

    # Move RMSD/Label column to a specific position (e.g., before the last few columns or specific index)
    # Original code inserted at 1482. Here we handle it dynamically or append to end.
    # To strictly follow original logic: insert before the last column if strictly needed, 
    # but usually labels are kept at the end.
    # Let's keep the logic of explicitly placing it if needed, or just renaming it properly.
    
    # Extract data series
    rmsd_data = merged_df.pop(target_col)
    
    # Insert at specific index (1482 as per original request, but let's be safe)
    # If dataframe is smaller than 1482 columns, append to end.
    insert_loc = 1482 if len(merged_df.columns) >= 1482 else len(merged_df.columns)
    merged_df.insert(insert_loc, "rmsd", rmsd_data)

    # 8. Save Result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    print(f"Merged Final Count: {len(merged_df)}")
    merged_df.to_pickle(output_path)
    print(f"✔ Successfully saved merged data to: {output_path}")

if __name__ == "__main__":
    merge_datasets()