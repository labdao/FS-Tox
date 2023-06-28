from rdkit import RDLogger
from rdkit import Chem


# Define a function to convert InChI to SMILES
def convert_to_smiles(inchi):

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")
    
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return 'InvalidInChI'  # Placeholder for invalid InChI
    smiles = Chem.MolToSmiles(mol)
    return smiles