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

# Based on alphafold/common/residue_constants.py

import copy
import dataclasses
import collections
import functools
import os
from typing import Dict, List, Mapping, Tuple

import numpy as np

"""Constants used in AlphaFold."""

# fmt:off
# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
RESIDUE_ATOMS = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3",
            "CH2", "N", "NE1", "O"],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O",
            "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}

# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
# in LEU, VAL and ARG can be resolved by using the 3d constellations of
# the 'ambiguous' atoms and their neighbours)
RESIDUE_ATOM_RENAMING_SWAPS = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
ATOM_TYPES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
]
ATOM_ORDER = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}
ATOM_TYPE_NUM = len(ATOM_TYPES)  # := 37.

# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
# fmt:off
RESTYPE_NAME_TO_ATOM14_NAMES = {
    "ALA": ["N", "CA", "C", "O", "CB",   "",   "",   "",   "",   "",   "",   "",   "",   ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ","NH1","NH2",   "",   "",   ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG","OD1","ND2",   "",   "",   "",   "",   "",   ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG","OD1","OD2",   "",   "",   "",   "",   "",   ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG",   "",   "",   "",   "",   "",   "",   "",   ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD","OE1","NE2",   "",   "",   "",   "",   ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD","OE1","OE2",   "",   "",   "",   "",   ""],
    "GLY": ["N", "CA", "C", "O",   "",   "",   "",   "",   "",   "",   "",   "",   "",   ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG","ND1","CD2","CE1","NE2",   "",   "",   "",   ""],
    "ILE": ["N", "CA", "C", "O", "CB","CG1","CG2","CD1",   "",   "",   "",   "",   "",   ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG","CD1","CD2",   "",   "",   "",   "",   "",   ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ",   "",   "",   "",   "",   ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE",   "",   "",   "",   "",   "",   ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG","CD1","CD2","CE1","CE2", "CZ",   "",   "",   ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD",   "",   "",   "",   "",   "",   "",   ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG",   "",   "",   "",   "",   "",   "",   "",   ""],
    "THR": ["N", "CA", "C", "O", "CB","OG1","CG2",   "",   "",   "",   "",   "",   "",   ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG","CD1","CD2","CE1","CE2", "CZ", "OH",   "",   ""],
    "VAL": ["N", "CA", "C", "O", "CB","CG1","CG2",   "",   "",   "",   "",   "",   "",   ""],
    "UNK": [ "",   "",  "",  "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   ""],
}
# fmt:on
# pylint: enable=line-too-long
# pylint: enable=bad-whitespace

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
RESTYPES = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
RESTYPE_ORDER = {restype: i for i, restype in enumerate(RESTYPES)}
RESTYPE_NUM = len(RESTYPES)  # := 20.
UNK_RESTYPE_INDEX = RESTYPE_NUM  # Catch-all index for unknown restypes.

RESTYPES_WITH_X = RESTYPES + ["X"]
RESTYPE_ORDER_WITH_X = {r: i for i, r in enumerate(RESTYPES_WITH_X)}

RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
RESTYPE_3TO1 = {v: k for k, v in RESTYPE_1TO3.items()}

# Define a restype name for all unknown residues.
UNK_RESTYPE = "UNK"

RESNAMES = [RESTYPE_1TO3[r] for r in RESTYPES] + [UNK_RESTYPE]
RESNAME_TO_IDX = {resname: i for i, resname in enumerate(RESNAMES)}


# The mapping here uses hhblits convention, so that B is mapped to D, J and O
# are mapped to X, U is mapped to C, and Z is mapped to E. Other than that the
# remaining 20 amino acids are kept in alphabetical order.
# There are 2 non-amino acid codes, X (representing any amino acid) and
# "-" representing a missing amino acid in an alignment.  The id for these
# codes is put at the end (20 and 21) so that they can easily be ignored if
# desired.
HHBLITS_AA_TO_ID = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
    "-": 21,
}

# Partial inversion of HHBLITS_AA_TO_ID.
ID_TO_HHBLITS_AA = {
    0: "A",
    1: "C",  # Also U.
    2: "D",  # Also B.
    3: "E",  # Also Z.
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",  # Includes J and O.
    21: "-",
}

# fmt:on

RESTYPES_WITH_X_AND_GAP = RESTYPES + ["X", "-"]
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
    RESTYPES_WITH_X_AND_GAP.index(ID_TO_HHBLITS_AA[i])
    for i in range(len(RESTYPES_WITH_X_AND_GAP))
)
RESTYPE_ORDER_WITH_X_AND_GAP = {
    k: i for i, k in enumerate(RESTYPES_WITH_X_AND_GAP)
}


def _make_standard_atom_mask() -> np.ndarray:
    """Returns [num_res_types, num_atom_types] mask array."""
    # +1 to account for unknown (all 0s).
    mask = np.zeros([RESTYPE_NUM + 1, ATOM_TYPE_NUM], dtype=np.int32)
    for i, restype_letter in enumerate(RESTYPES):
        resname = RESTYPE_1TO3[restype_letter]
        atom_names = RESIDUE_ATOMS[resname]
        for atom_name in atom_names:
            atom_type = ATOM_ORDER[atom_name]
            mask[i, atom_type] = 1
    return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()


def _make_restype_atom_mask(atom_num: int) -> np.ndarray:
    """Mask of which atoms are present for which residue type in atom14."""
    assert atom_num in {14, 37}
    atom_mask = []
    for restype in RESTYPES:
        resname = RESTYPE_1TO3[restype]
        if atom_num == 14:
            src_list = RESTYPE_NAME_TO_ATOM14_NAMES[resname]
            atom_order_map = {a: i for i, a in enumerate(src_list) if a}
        else:
            src_list = RESIDUE_ATOMS[resname]
            atom_order_map = ATOM_ORDER
        _mask = [0] * atom_num
        for atom_name in src_list:
            if atom_name in atom_order_map:
                _mask[atom_order_map[atom_name]] = 1
        atom_mask.append(_mask)
    # Add dummy mask for restype 'UNK'
    atom_mask.append([0] * atom_num)
    return np.array(atom_mask, dtype=bool)


def _make_restype_atom_conversion(atom_num: int) -> np.ndarray:
    """Map from atom14(37) to atom37(14) per residue type."""
    assert atom_num in {14, 37}
    atoms_map = []  # mapping (restype, atom14(37)) --> atom37(14)
    atom_order = ATOM_ORDER
    for restype in RESTYPES:
        resname = RESTYPE_1TO3[restype]
        if atom_num == 14:
            src_list = RESTYPE_NAME_TO_ATOM14_NAMES[resname]
            tgt = {name: ATOM_ORDER.get(name, 0) for name in src_list}
        else:
            src_list = ATOM_TYPES
            tgt_list = RESTYPE_NAME_TO_ATOM14_NAMES[resname]
            tgt = {name: i for i, name in enumerate(tgt_list)}
        atoms_map.append([tgt.get(name, 0) for name in src_list])
    # Add dummy mapping for restype 'UNK'
    atoms_map.append([0] * atom_num)
    return np.array(atoms_map, dtype=np.int32)


RESTYPE_ATOM14_TO_ATOM37 = _make_restype_atom_conversion(14)  # shape (21, 14)
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom_conversion(37)  # shape (21, 37)
RESTYPE_ATOM14_MASK = _make_restype_atom_mask(14)  # shape (21, 14)
RESTYPE_ATOM37_MASK = _make_restype_atom_mask(37)  # shape (21, 37)


def _get_atom_list(restype):
    """Get atom14 list of the restype"""
    resname = RESTYPE_1TO3[restype]
    atom_list = RESTYPE_NAME_TO_ATOM14_NAMES[resname]
    return atom_list


"""Atom distances (bonds and radius)
"""

# Distance from one CA to next CA [trans configuration: omega = 180].
CA_CA = 3.80209737096

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
VAN_DER_WAALS_RADIUS = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}

# (the first letter of the atom name is the element type). Shape (37)
ATOM37_RADIUS = np.array(
    [VAN_DER_WAALS_RADIUS[name[0]] for name in ATOM_TYPES], dtype=np.float32
)


def _make_atom14_radius() -> np.ndarray:
    atom14_radius = []
    for restype in RESTYPES:
        atom_list = _get_atom_list(restype)
        atom_list = [ATOM_ORDER.get(a, 0) for a in atom_list]
        atom14_radius.append([ATOM37_RADIUS[i] for i in atom_list])
    atom14_radius.append([0] * 14)
    return np.array(atom14_radius, np.float32) * RESTYPE_ATOM14_MASK


RESTYPE_ATOM14_RADIUS = _make_atom14_radius()  # shape (21, 14)
RESTYPE_ATOM37_RADIUS = (
    np.repeat(ATOM37_RADIUS[None], 21, axis=0) * RESTYPE_ATOM37_MASK
)  # (21, 37)


Bond = collections.namedtuple(
    "Bond", ["atom1_name", "atom2_name", "length", "stddev"]
)
BondAngle = collections.namedtuple(
    "BondAngle",
    ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
)


@functools.lru_cache(maxsize=None)
def _load_stereo_chemical_props() -> Tuple[
    Dict[str, List[Bond]],
    Dict[str, List[Bond]],
    Dict[str, List[BondAngle]],
]:
    """Load stereo_chemical_props.txt into a nice structure.

    Load literature values for bond lengths and bond angles and translate
    bond angles into the length of the opposite edge of the triangle
    ("residue_virtual_bonds").

    Returns:
      residue_bonds: Dict that maps resname -> list of Bond tuples.
      residue_virtual_bonds: Dict that maps resname -> list of Bond tuples.
      residue_bond_angles: Dict that maps resname -> list of BondAngle tuples.
    """
    stereo_chemical_props_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stereo_chemical_props.txt"
    )
    with open(stereo_chemical_props_path, "rt") as f:
        stereo_chemical_props = f.read()
    lines_iter = iter(stereo_chemical_props.splitlines())
    # Load bond lengths.
    residue_bonds = {}
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, length, stddev = line.split()
        atom1, atom2 = bond.split("-")
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        residue_bonds[resname].append(
            Bond(atom1, atom2, float(length), float(stddev))
        )
    residue_bonds["UNK"] = []

    # Load bond angles.
    residue_bond_angles = {}
    next(lines_iter)  # Skip empty line.
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == "-":
            break
        bond, resname, angle_degree, stddev_degree = line.split()
        atom1, atom2, atom3 = bond.split("-")
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        residue_bond_angles[resname].append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                float(angle_degree) / 180.0 * np.pi,
                float(stddev_degree) / 180.0 * np.pi,
            )
        )
    residue_bond_angles["UNK"] = []

    def make_bond_key(atom1_name, atom2_name):
        """Unique key to lookup bonds."""
        return "-".join(sorted([atom1_name, atom2_name]))

    # Translate bond angles into distances ("virtual bonds").
    residue_virtual_bonds = {}
    for resname, bond_angles in residue_bond_angles.items():
        # Create a fast lookup dict for bond lengths.
        bond_cache = {}
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        residue_virtual_bonds[resname] = []
        for ba in bond_angles:
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

            # Compute distance between atom1 and atom3 using the law of cosines
            # c^2 = a^2 + b^2 - 2ab*cos(gamma).
            gamma = ba.angle_rad
            length = np.sqrt(
                bond1.length**2
                + bond2.length**2
                - 2 * bond1.length * bond2.length * np.cos(gamma)
            )

            # Propagation of uncertainty assuming uncorrelated errors.
            dl_outer = 0.5 / length
            dl_dgamma = (
                2 * bond1.length * bond2.length * np.sin(gamma)
            ) * dl_outer
            dl_db1 = (
                2 * bond1.length - 2 * bond2.length * np.cos(gamma)
            ) * dl_outer
            dl_db2 = (
                2 * bond2.length - 2 * bond1.length * np.cos(gamma)
            ) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev) ** 2
                + (dl_db1 * bond1.stddev) ** 2
                + (dl_db2 * bond2.stddev) ** 2
            )
            residue_virtual_bonds[resname].append(
                Bond(ba.atom1_name, ba.atom3name, length, stddev)
            )

    return residue_bonds, residue_virtual_bonds, residue_bond_angles


(
    RESIDUE_BONDS,
    RESIDUE_VIRTUAL_BONDS,
    RESIDUE_BOND_ANGLES,
) = _load_stereo_chemical_props()


def _make_bond_lengths(atom_num: int):
    """Compute upper and lower bounds for bonds to assess violations."""
    bond_lengths = np.zeros([21, atom_num, atom_num], dtype=np.float32)
    bond_stddevs = np.zeros([21, atom_num, atom_num], dtype=np.float32)

    for restype, restype_letter in enumerate(RESTYPES):
        resname = RESTYPE_1TO3[restype_letter]
        if atom_num == 37:
            atom_list = ATOM_TYPES
        else:
            atom_list = RESTYPE_NAME_TO_ATOM14_NAMES[resname]

        for b in RESIDUE_BONDS[resname] + RESIDUE_VIRTUAL_BONDS[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            bond_lengths[restype, atom1_idx, atom2_idx] = b.length
            bond_lengths[restype, atom2_idx, atom1_idx] = b.length
            bond_stddevs[restype, atom1_idx, atom2_idx] = b.stddev
            bond_stddevs[restype, atom2_idx, atom1_idx] = b.stddev

    return bond_lengths, bond_stddevs


# zero if there is not a bond, shape (21, 14/37, 14/37)
ATOM14_BOND_LENGTHS, ATOM14_BOND_STDDEVS = _make_bond_lengths(14)
ATOM37_BOND_LENGTHS, ATOM37_BOND_STDDEVS = _make_bond_lengths(37)


def make_dists_bounds(
    atom_num: int,
    overlap_tolerance: float = 1.5,
    bond_length_tolerance_factor: float = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upper and lower bounds for bonds to assess violations."""
    assert atom_num in {14, 37}
    aatype = np.arange(21, dtype=np.int32)
    if atom_num == 14:
        atom_radius = RESTYPE_ATOM14_RADIUS[aatype]
        atom_mask = RESTYPE_ATOM14_MASK[aatype].astype(np.float32)
        bond_lengths = ATOM14_BOND_LENGTHS
        bond_stddevs = ATOM14_BOND_STDDEVS
    else:
        atom_radius = RESTYPE_ATOM37_RADIUS
        atom_mask = RESTYPE_ATOM37_MASK[aatype].astype(np.float32)
        bond_lengths = ATOM37_BOND_LENGTHS
        bond_stddevs = ATOM37_BOND_STDDEVS

    bond_mask = (bond_lengths > 0).astype(np.float32)
    atom_dist = atom_radius[:, :, None] + atom_radius[:, None, :]
    atom_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
    atom_mask *= 1 - np.eye(atom_num, dtype=np.float32)[None]
    bond_lower_bounds, bond_upper_bounds = (
        bond_lengths - bond_stddevs * bond_length_tolerance_factor,
        bond_lengths + bond_stddevs * bond_length_tolerance_factor,
    )
    lower_bounds = (1 - bond_mask) * (
        atom_dist - overlap_tolerance
    ) * atom_mask + bond_mask * bond_lower_bounds
    upper_bounds = (
        1 - bond_mask
    ) * atom_mask * 1e10 + bond_mask * bond_upper_bounds
    dist_stddevs = bond_mask * bond_stddevs

    return lower_bounds, upper_bounds, dist_stddevs


"""Rigid groups
"""
# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
CHI_ANGLES_ATOMS = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
CHI_ANGLES_MASK = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
CHI_PI_PERIODIC = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]


# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# CHI_ANGLES_ATOMS above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
RIGID_GROUP_ATOM_POSITIONS = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}


def _make_rigid_transformation_4x4(ex, ey, translation):
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # Normalize ex.
    ex_normalized = ex / np.linalg.norm(ex)

    # make ey perpendicular to ex
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # compute ez as cross product
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack(
        [ex_normalized, ey_normalized, eznorm, translation]
    ).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m


def _make_rigid_group_frames():
    rigid_group_frames = np.zeros([21, 8, 4, 4])

    for restype, restype_letter in enumerate(RESTYPES):
        resname = RESTYPE_1TO3[restype_letter]
        atom_positions = {
            name: np.array(pos)
            for name, _, pos in RIGID_GROUP_ATOM_POSITIONS[resname]
        }

        # backbone to backbone is the identity transform
        rigid_group_frames[restype, 0, :, :] = np.eye(4)
        # pre-omega-frame to backbone (currently dummy identity matrix)
        rigid_group_frames[restype, 1, :, :] = np.eye(4)

        # phi-frame to backbone (C^-N-CA, N-CA-C)
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["N"] - atom_positions["CA"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=atom_positions["N"],
        )
        rigid_group_frames[restype, 2, :, :] = mat

        # psi-frame to backbone (N-CA-C, CA-C-N')
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"] - atom_positions["N"],  # N' - C
            translation=atom_positions["C"],
        )
        rigid_group_frames[restype, 3, :, :] = mat

        # chi1-frame to backbone
        if CHI_ANGLES_MASK[restype][0]:
            base_atom_positions = [
                atom_positions[name] for name in CHI_ANGLES_ATOMS[resname][0]
            ]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            rigid_group_frames[restype, 4, :, :] = mat

        # chi2-frame to chi1-frame
        # chi3-frame to chi2-frame
        # chi4-frame to chi3-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for chi_idx in range(1, 4):
            if CHI_ANGLES_MASK[restype][chi_idx]:
                axis_end_atom_name = CHI_ANGLES_ATOMS[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=np.array([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                rigid_group_frames[restype, 4 + chi_idx, :, :] = mat

    return rigid_group_frames


@dataclasses.dataclass(frozen=True)
class AtomRigidGroup:
    atom_to_rigid_group: np.ndarray  # Shape [21, 14/37]
    atom_mask: np.ndarray  # Shape [21, 14/37]
    atom_positions: np.ndarray  # Shape [21, 14/37, 3]

    def to_tensors(self):
        import torch

        return (
            torch.from_numpy(self.atom_to_rigid_group).long(),
            torch.from_numpy(self.atom_mask).float(),
            torch.from_numpy(self.atom_positions).float(),
        )


def _make_rigid_group():
    def _make(n):
        return np.zeros((21, n)), np.zeros((21, n)), np.zeros((21, n, 3))

    atom37_to_rigid_group, atom37_mask, atom37_positions = _make(37)
    atom14_to_rigid_group, atom14_mask, atom14_positions = _make(14)

    for restype, restype_letter in enumerate(RESTYPES):
        resname = RESTYPE_1TO3[restype_letter]
        for atomname, group_idx, pos in RIGID_GROUP_ATOM_POSITIONS[resname]:
            atomtype = ATOM_ORDER[atomname]
            atom37_to_rigid_group[restype, atomtype] = group_idx
            atom37_mask[restype, atomtype] = 1
            atom37_positions[restype, atomtype, :] = pos

            atom14idx = RESTYPE_NAME_TO_ATOM14_NAMES[resname].index(atomname)
            atom14_to_rigid_group[restype, atom14idx] = group_idx
            atom14_mask[restype, atom14idx] = 1
            atom14_positions[restype, atom14idx, :] = pos

    return (
        AtomRigidGroup(atom37_to_rigid_group, atom37_mask, atom37_positions),
        AtomRigidGroup(atom14_to_rigid_group, atom14_mask, atom14_positions),
    )


RIGID_GROUP_FRAMES = _make_rigid_group_frames()  # Shape [21, 8, 4, 4]
ATOM37_RIGID_GROUP, ATOM14_RIGID_GROUP = _make_rigid_group()


def _make_restype_rigidgroup_base_atom37_idx():
    """Create Map from rigidgroups to atom37 indices."""
    # Create an array with the atom names.
    # shape (num_restypes, num_rigidgroups, 3_atoms): (21, 8, 3)
    base_atom_names = np.full([21, 8, 3], "", dtype=object)

    # 0: backbone frame
    base_atom_names[:, 0, :] = ["C", "CA", "N"]

    # 3: 'psi-group'
    base_atom_names[:, 3, :] = ["CA", "C", "O"]

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(RESTYPES):
        resname = RESTYPE_1TO3[restype_letter]
        for chi_idx in range(4):
            if CHI_ANGLES_MASK[restype][chi_idx]:
                atom_names = CHI_ANGLES_ATOMS[resname][chi_idx]
                base_atom_names[restype, chi_idx + 4, :] = atom_names[1:]

    # Translate atom names into atom37 indices.
    lookuptable = copy.deepcopy(ATOM_ORDER)
    lookuptable[""] = 0
    rigidgroup = np.vectorize(lambda x: lookuptable[x])(base_atom_names)

    return rigidgroup


# rigid group to atom37 idx
RESTYPE_RIGIDGROUP_BASE_ATOM37_IDX = _make_restype_rigidgroup_base_atom37_idx()

# Create mask for existing rigid groups.
RESTYPE_RIGIDGROUP_MASK = np.zeros([21, 8], dtype=np.float32)
RESTYPE_RIGIDGROUP_MASK[:, 0] = 1
RESTYPE_RIGIDGROUP_MASK[:, 3] = 1
RESTYPE_RIGIDGROUP_MASK[:20, 4:] = CHI_ANGLES_MASK


def _make_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types."""
    chi_atom_indices = []
    for restype_letter in RESTYPES:
        res_name = RESTYPE_1TO3[restype_letter]
        atom_indices = [(0, 0, 0, 0)] * 4
        for group_idx, group_atoms in enumerate(CHI_ANGLES_ATOMS[res_name]):
            atom_indices[group_idx] = [ATOM_ORDER[atom] for atom in group_atoms]
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    chi_atom_mask = np.array(tuple(CHI_ANGLES_MASK) + ([0.0, 0.0, 0.0, 0.0],))

    return np.array(chi_atom_indices, dtype=np.int64), chi_atom_mask


# (21, 4, 4), (21, 4)
CHI_ATOM_INDICES, CHI_ATOM_MASK = _make_chi_atom_indices()


# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
BETWEEN_RES_BOND_LENGTH_C_N = [1.329, 1.341]
BETWEEN_RES_BOND_LENGTH_STDDEV_C_N = [0.014, 0.016]

# Between-residue cos_angles.
BETWEEN_RES_COS_ANGLES_C_N_CA = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
BETWEEN_RES_COS_ANGLES_CA_C_N = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995

"""Alternative positions
"""


def _make_renaming_matrices_and_mask() -> Tuple[np.ndarray, np.ndarray]:
    """Make renaming transformation matrics for atom14 representations."""
    resname_list = [RESTYPE_1TO3[res] for res in RESTYPES] + ["UNK"]
    _renaming_matrices = np.repeat(np.eye(14)[None], 21, axis=0)
    _restype_atom14_is_amb = np.zeros((21, 14))
    for res_idx, resname in enumerate(resname_list):
        if resname in RESIDUE_ATOM_RENAMING_SWAPS:
            atom_list = RESTYPE_NAME_TO_ATOM14_NAMES[resname]
            for src, tgt in RESIDUE_ATOM_RENAMING_SWAPS[resname].items():
                src = atom_list.index(src)
                tgt = atom_list.index(tgt)
                _renaming_matrices[res_idx, src, tgt] = 1
                _renaming_matrices[res_idx, tgt, src] = 1
                _renaming_matrices[res_idx, src, src] = 0
                _renaming_matrices[res_idx, tgt, tgt] = 0
                _restype_atom14_is_amb[res_idx, src] = 1
                _restype_atom14_is_amb[res_idx, tgt] = 1

    return _renaming_matrices, _restype_atom14_is_amb


RENAMING_MATRICES, RESTYPE_ATOM14_IS_AMB = _make_renaming_matrices_and_mask()


def _make_alt_rigidgroup() -> Tuple[np.ndarray, np.ndarray]:
    _restype_rigidgroup_is_ambiguous = np.zeros((21, 8))
    _restype_rigidgroup_rots = np.tile(np.eye(3), [21, 8, 1, 1])
    for resname, _alt_atoms in RESIDUE_ATOM_RENAMING_SWAPS.items():
        restype = RESTYPE_ORDER[RESTYPE_3TO1[resname]]
        chi_idx = int(sum(CHI_ANGLES_MASK[restype]) - 1)
        _restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        _restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        _restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    return _restype_rigidgroup_is_ambiguous, _restype_rigidgroup_rots


RESTYPE_RIGIDGROUP_IS_AMB, RESTYPE_RIGIDGROUP_ROTS = _make_alt_rigidgroup()
