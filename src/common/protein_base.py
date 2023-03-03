# Modified by zw2x

# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import io
import logging
import dataclasses
from typing import Any, Mapping, Optional, Dict, Tuple, Sequence

import numpy as np
from Bio.PDB import PDBParser
from Bio import PDB
from Bio.Data import SCOPData

from . import rc

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # rc.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]



def from_pdb_string(
    pdb_str: str,
    chain_id: Optional[str] = None,
    return_auth_chain_ids: bool = False,
    ignore_insertion: bool = True,
) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    # if len(models) != 1:
    #     raise ValueError(
    #         f"Only single model PDBs are supported. Found {len(models)} models."
    #     )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                if ignore_insertion:
                    continue
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported.\n"
                    f"Residue {res}."
                )
            if res.id[0] != " ":
                continue
            res_shortname = rc.RESTYPE_3TO1.get(res.resname, "X")
            restype_idx = rc.RESTYPE_ORDER.get(res_shortname, rc.RESTYPE_NUM)
            pos = np.zeros((rc.ATOM_TYPE_NUM, 3))
            mask = np.zeros((rc.ATOM_TYPE_NUM,))
            res_b_factors = np.zeros((rc.ATOM_TYPE_NUM,))
            for atom in res:
                if atom.name not in rc.ATOM_TYPES:
                    continue
                pos[rc.ATOM_ORDER[atom.name]] = atom.coord
                mask[rc.ATOM_ORDER[atom.name]] = 1.0
                res_b_factors[rc.ATOM_ORDER[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    if return_auth_chain_ids:
        chain_ids = [c.encode() for c in chain_ids]
        chain_index = np.array(chain_ids, dtype=object)
    else:
        unique_chain_ids = np.unique(chain_ids)
        chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
        chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def to_pdb(
    prot: Protein,
    pdb_chain_ids: Dict[int, str] = PDB_CHAIN_IDS,
    ignore_chain_end: bool = False,
) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = rc.RESTYPES + ["X"]
    res_1to3 = lambda r: rc.RESTYPE_1TO3.get(restypes[r], "UNK")
    atom_types = rc.ATOM_TYPES

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > rc.RESTYPE_NUM):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = pdb_chain_ids[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            if not ignore_chain_end:
                pdb_lines.append(
                    _chain_end(
                        atom_index,
                        res_1to3(aatype[i - 1]),
                        chain_ids[chain_index[i - 1]],
                        residue_index[i - 1],
                    )
                )
                last_chain_index = chain_index[i]
                atom_index += 1  # Atom index increases at the TER symbol.
            else:
                last_chain_index = chain_index[i]

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return rc.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A protein instance.
    """
    fold_output = result["structure_module"]

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"])
    else:
        chain_index = np.zeros_like(
            _maybe_remove_leading_dim(features["aatype"])
        )

    if b_factors is None:
        b_factors = np.zeros_like(fold_output["final_atom_mask"])

    return Protein(
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=fold_output["final_atom_positions"],
        atom_mask=fold_output["final_atom_mask"],
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def _check_residue_distances(
    all_positions: np.ndarray,
    all_positions_mask: np.ndarray,
    max_ca_ca_distance: float,
):
    """Checks if the distance between unmasked neighbor residues is ok."""
    ca_position = rc.ATOM_ORDER["CA"]
    prev_is_unmasked = False
    prev_calpha = None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
        this_is_unmasked = bool(mask[ca_position])
        if this_is_unmasked:
            this_calpha = coords[ca_position]
            if prev_is_unmasked:
                distance = np.linalg.norm(this_calpha - prev_calpha)
                if distance > max_ca_ca_distance:
                    raise ValueError(
                        "The distance between residues %d and %d is %f > limit"
                        " %f." % (i, i + 1, distance, max_ca_ca_distance)
                    )
            prev_calpha = this_calpha
        prev_is_unmasked = this_is_unmasked


def _get_res_id(p) -> Tuple[str, int, str]:
    return (p.hetflag, p.position.residue_number, p.position.insertion_code)


def _get_sequence(chain: PDB.Chain.Chain, chain_mapping, num_res: int) -> str:
    _pos = [chain_mapping[i] for i in range(num_res)]
    _res = [chain[_get_res_id(p)] for p in _pos if not p.is_missing]
    _seq = "".join([rc.RESTYPE_3TO1.get(res.resname, "X") for res in _res])
    return _seq


X_INDEX = rc.RESTYPE_ORDER_WITH_X["X"]



def _get_residue_feature(res: PDB.Residue.Residue) -> Dict[str, np.ndarray]:
    _atom_num = rc.ATOM_TYPE_NUM
    pos = np.zeros([_atom_num, 3], dtype=np.float32)
    mask = np.zeros([_atom_num], dtype=np.float32)
    bfactor = np.zeros([_atom_num], dtype=np.float32)
    for atom in res.get_atoms():
        atom_name = atom.get_name()
        x, y, z = atom.get_coord()
        if atom_name in rc.ATOM_ORDER.keys():
            pos[rc.ATOM_ORDER[atom_name]] = [x, y, z]
            mask[rc.ATOM_ORDER[atom_name]] = 1.0
            bfactor[rc.ATOM_ORDER[atom_name]] = atom.bfactor
        elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
            # Put the coordinates of the selenium atom in the sulphur
            # column.
            pos[rc.ATOM_ORDER["SD"]] = [x, y, z]
            mask[rc.ATOM_ORDER["SD"]] = 1.0
            bfactor[rc.ATOM_ORDER["SD"]] = atom.bfactor

    # Fix naming errors in arginine residues where NH2 is incorrectly
    # assigned to be closer to CD than NH1.
    cd = rc.ATOM_ORDER["CD"]
    nh1 = rc.ATOM_ORDER["NH1"]
    nh2 = rc.ATOM_ORDER["NH2"]
    if (
        res.get_resname() == "ARG"
        and all(mask[atom_index] for atom_index in (cd, nh1, nh2))
        and (
            np.linalg.norm(pos[nh1] - pos[cd])
            > np.linalg.norm(pos[nh2] - pos[cd])
        )
    ):
        pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
        mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()
        bfactor[nh1], bfactor[nh2] = (bfactor[nh2].copy(), bfactor[nh1].copy())

    return {"pos": pos, "mask": mask, "bfactor": bfactor}


def from_aatype_to_sequence(aatype: np.ndarray) -> str:
    seq = "".join(rc.RESTYPES_WITH_X_AND_GAP[i] for i in aatype)
    return seq


X_IDX = rc.RESTYPE_ORDER_WITH_X_AND_GAP["X"]


def from_sequence_to_aatype(sequence: str) -> np.ndarray:
    aatype = np.array(
        [rc.RESTYPE_ORDER_WITH_X_AND_GAP.get(aa, X_IDX) for aa in sequence],
        dtype=np.uint8,
    )
    return aatype


def from_sequence_to_hhblits_aatype(sequence: str) -> np.ndarray:
    aatype = np.array(
        [rc.HHBLITS_AA_TO_ID[aa] for aa in sequence], dtype=np.uint8
    )
    return aatype


def from_hhblits_aatype_to_our_aatype(hhblits_aatype):
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    aatype = np.take(new_order_list, hhblits_aatype.astype(np.int32), axis=0)
    return aatype
