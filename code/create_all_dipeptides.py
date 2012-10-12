import os

amino_acids = ["A","S","T","Y","F","C","M","H","D","E","Q","W","R","T","Y","I","N","L","K"]

for aa0 in amino_acids:
    for aa1 in amino_acids:
        cmd = """pymol -qc ~/src/pymd/KyleCode/CreateChain.py  -- 'b%s%su' ./%s%s.pdb"""%(aa0,aa1,aa0,aa1)
        os.system(cmd)