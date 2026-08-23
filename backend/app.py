import streamlit as st
import torch
import numpy as np
import sys
import os

# ===== IMPORT MODEL =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import PredictModel

# ===== LOAD MODEL =====
device = torch.device("cpu")

model = PredictModel(
    num_layers=6,
    d_model=512,
    dff=512,
    num_heads=8,
    vocab_size=18
)

model_path = os.path.join(os.path.dirname(__file__), "..", "best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== RDKit IMPORT =====
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# ===== FUNCTIONS =====

def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.array(fp)
    return torch.tensor(arr, dtype=torch.float32)

def compute_aac(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq = sequence.upper()

    length = len(seq)
    aac = []

    for aa in amino_acids:
        count = seq.count(aa)
        aac.append(count / length if length > 0 else 0)

    return torch.tensor(aac, dtype=torch.float32)

SMILES_VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=#()[]+-/@"

def smiles_to_tokens(smiles, max_len=50):
    tokens = [(ord(c) % 17) + 1 for c in smiles]

    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    return torch.tensor(tokens)

# ===== UI =====

st.title("🧬 ABFormer Predictor")
st.write("Enter biological inputs to predict therapeutic activity")

with st.form("prediction_form"):

    payload_smiles = st.text_input("Payload SMILES")
    linker_smiles = st.text_input("Linker SMILES")

    heavy_chain = st.text_area("Antibody Heavy Chain Sequence")
    light_chain = st.text_area("Antibody Light Chain Sequence")
    antigen = st.text_area("Antigen Sequence")

    dar = st.number_input("DAR value", min_value=0.0, step=0.1)

    submit = st.form_submit_button("Predict")


# 👉 THIS MUST BE OUTSIDE THE FORM
if submit:

    payload_smiles = payload_smiles.strip()
    linker_smiles = linker_smiles.strip()
    heavy_chain = heavy_chain.strip()
    light_chain = light_chain.strip()
    antigen = antigen.strip()

    # ===== VALIDATION =====
    if payload_smiles == "" or linker_smiles == "":
        st.error("Please enter SMILES inputs")
        st.stop()

    if heavy_chain == "" or light_chain == "" or antigen == "":
        st.error("Please enter all sequences")
        st.stop()

    # ===== DEBUG (ADD THIS TEMPORARILY) =====
    st.write("Running prediction...")

    # ===== PROCESS =====
    x1 = smiles_to_tokens(payload_smiles).unsqueeze(0)
    x2 = smiles_to_tokens(linker_smiles).unsqueeze(0)

    x1_maccs = smiles_to_maccs(payload_smiles)
    x2_maccs = smiles_to_maccs(linker_smiles)

    if x1_maccs is None or x2_maccs is None:
        st.error("Invalid SMILES input")
        st.stop()

    x1_maccs = x1_maccs.unsqueeze(0)
    x2_maccs = x2_maccs.unsqueeze(0)

    aac1 = compute_aac(heavy_chain).unsqueeze(0)
    aac2 = compute_aac(light_chain).unsqueeze(0)
    aac3 = compute_aac(antigen).unsqueeze(0)

    t1 = torch.zeros((1, 2592))
    t2 = torch.zeros((1, 1280))
    t3 = torch.zeros((1, 1280))
    t4 = torch.tensor([[dar]])

    with torch.no_grad():
        output = model(
            x1, x1_maccs,
            x2, x2_maccs,
            t1, t2, t3,
            aac1, aac2, aac3,
            t4
        )

        prob = torch.sigmoid(output).item()

    # ===== OUTPUT =====
    st.success(f"Prediction: {prob:.4f}")
    st.write("Result:", "Active ✅" if prob > 0.5 else "Inactive ❌")