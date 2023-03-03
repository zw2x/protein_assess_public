"""
Lafita, Aleix, et al. 
    Assessment of protein assembly prediction in CASP12.
    Proteins: Structure, Function, and Bioinformatics 86 (2018): 247-256.
"""

import io
import logging
import dataclasses
from itertools import product
from typing import Any, Mapping, Optional, Dict, Tuple, Sequence
from tqdm import tqdm

import numpy as np
from Bio.PDB import PDBParser
from Bio import PDB

def parse_atoms_from_pdb(
    pdb_str: str,
    chain_id: Optional[str] = None,
    return_auth_chain_ids: bool = False,
    ignore_insertion: bool = True,
):
    """Takes a PDB string and returns a dictionary

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    model = models[0]

    all_atoms = []
    aatype = []
    residue_ids = []
    chain_ids = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                if ignore_insertion:
                    continue
                raise ValueError()
            if res.id[0] != " ":
                continue
            _atoms = []
            for atom in res:
                _atom = {
                    "atom_type": atom.name,
                    "coord": atom.coord,
                    "bfactor": atom.bfactor,
                }
                _atoms.append(_atom)
            all_atoms.append(_atoms)
            aatype.append(res.resname)
            residue_ids.append(res.id)
            chain_ids.append(chain.id)

    prot_dict = {
        "all_atoms": all_atoms,
        "aatype": aatype,
        "res_ids": residue_ids,
        "chain_ids": chain_ids,
    }

    return prot_dict

def _compute_residue_pair(atoms1, atoms2):
    heavy_atom_dists = []
    for atom1, atom2 in product(atoms1, atoms2):
        atom1_type = atom1["atom_type"]
        atom2_type = atom2["atom_type"]
        if atom1_type[0] == "H" or atom2_type[0] == "H":
            continue
        dist = np.sqrt(np.sum((atom1["coord"] - atom2["coord"]) ** 2))
        heavy_atom_dists.append((f"{atom1_type}:{atom2_type}", dist))

    rets = {
        "heavy_atom_dist": min([v[1] for v in heavy_atom_dists])
    }

    return rets

def find_residue_contacts(
    prot_dict,
    compute_interface: bool = True,
    heavy_dist_cutoff: float = 5.0,
    clash_dist_cutoff: float = 3.0,
    cbcb_dist_cutoff: float = 8.0,
    caca_dist_cutoff: float = 8.0,
):
    all_atoms = prot_dict["all_atoms"]
    num_res = len(all_atoms)
    heavy_atom_contacts = {}
    interface_residues = set()
    for i, j in product(range(num_res), range(num_res)):
        if i > j:
            continue
        chain_i, chain_j = prot_dict["chain_ids"][i], prot_dict["chain_ids"][j]
        if compute_interface and chain_i == chain_j:
            continue
        atoms_i, atoms_j = prot_dict["all_atoms"][i], prot_dict["all_atoms"][j]
        resid_i, resid_j = prot_dict["res_ids"][i], prot_dict["res_ids"][j]
        _pair = _compute_residue_pair(atoms_i, atoms_j)
        if _pair["heavy_atom_dist"] <= heavy_dist_cutoff:
            _pair_id = (resid_i,resid_j, chain_i, chain_j)
            heavy_atom_contacts[_pair_id] = _pair["heavy_atom_dist"]
            interface_residues.add((resid_i, chain_i))
            interface_residues.add((resid_j, chain_j))

    return heavy_atom_contacts, interface_residues

def compute_contact_metrics(ref_contacts, model_contacts):
    correct_contacts = set(ref_contacts.keys()) & set(model_contacts.keys())
    num_correct = len(correct_contacts)
    num_ref = len(ref_contacts)
    num_model = len(model_contacts)

    prec = num_correct / num_model if num_model > 0 else 0
    recall = num_correct / num_ref if num_ref > 0 else 0
    ics = 2 * (prec * recall) / (prec + recall) if prec > 0 and recall > 0 else 0

    return {
        "ics": ics,
        "precision": prec,
        "recall": recall,
    }

def compute_site_metrics(ref_sites, model_sites):
    correct_sites = set(ref_sites) & set(model_sites)
    common_sites = set(ref_sites) | set(model_sites)

    num_common = len(common_sites)

    ips = len(correct_sites) / num_common if num_common > 0 else 0

    return {
        "ips": ips
    }
