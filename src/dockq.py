import logging
import dataclasses
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Union, Tuple, Optional
from collections import defaultdict, OrderedDict

import numpy as np

from src.common import protein_base, aligner

logger = logging.getLogger(__file__)


@dataclasses.dataclass(frozen=True)
class DockQResult:
    fnat: float
    fnonnat: float
    irms: float
    lrms: float
    dockq: float
    # msg: str


class DockQRunner:
    def __init__(self, binary_path: str):
        super().__init__()

        self.binary_path = binary_path
        self.fix_numbering_binary_path = str(
            Path(binary_path).parent.joinpath("scripts", "fix_numbering.pl")
        )

    def read_chain_ids(self, model_path: Path):
        model_prot, model_seqs = read_pdb_file(model_path)
        return sorted(model_seqs.keys())

    def fix_numbering(
        self, model: Path, native: Path, fixed_model: Path
    ):
        chain_ids = fix_model(
            model,
            native,
            fixed_model,
        )

        cmd = [
            self.fix_numbering_binary_path, str(fixed_model), str(native),
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        output_str = output.decode()

        retcode = process.wait()
        if retcode:
            print(output_str)
            print(stderr)
            raise RuntimeError("Error during running fix_numbering.pl")

        _output_path = Path(str(fixed_model) + ".fixed")
        assert _output_path.exists()
        shutil.move(_output_path, fixed_model)

        return chain_ids

    def run(
        self,
        model: Path,
        native: Path,
        lig_id: str,
        rec_id: str,
        use_needle: bool = True,
    ) -> Optional[DockQResult]:
        """
        Note:
            Merge two chains (e.g. A and B) into ligand, pass 'A B' as lig_id
        """
        # fmt:off
        cmd = [
            self.binary_path,
            str(model),
            str(native),
            '-native_chain1', lig_id, '-model_chain1', lig_id,
            '-native_chain2', rec_id, '-model_chain2', rec_id,
        ]
        if not use_needle:
            cmd.append("-no_needle")
        # fmt:on
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        retcode = process.wait()

        output_str = output.decode()
        if retcode:
            if "length of native is zero" in output_str:
                logger.warning(
                    "###########################                           \n"
                    "Error during running DockQ:                           \n"
                    f"{output_str.strip()}                                 \n"
                    "--------------------------                            \n"
                    "the reason may be that the common_interface defined in\n"
                    "DockQ is empty                                        \n"
                    "##########################                              "
                )
            else:
                raise RuntimeError(output_str)

        result = _parse_result(output_str)

        if result is None:
            logger.warning(f"{cmd}\nOutput : {output_str}")

        return result, cmd


def _parse_result(output_str: str) -> Optional[DockQResult]:
    def _get_score(line, must_be_positive: bool = True):
        score = float(line.split()[1])
        if score < 0 and must_be_positive:
            logger.warning(f"Score should be positive: {score}, {line}.")
        return score

    result = {}
    lines = []
    for line in output_str.split("\n"):
        if line.startswith("Fnat"):
            result["fnat"] = _get_score(line)
        elif line.startswith("Fnonnat"):
            result["fnonnat"] = _get_score(line)
        elif line.startswith("iRMS"):
            result["irms"] = _get_score(line)
        elif line.startswith("LRMS"):
            result["lrms"] = _get_score(line)
        elif line.startswith("DockQ"):
            result["dockq"] = _get_score(line)
        lines.append(line)

    if "dockq" not in result:
        result = None
    else:
        result = DockQResult(**result)
    return result

def read_pdb_file(input_path, return_auth_chain_ids: bool = True):
    with open(input_path, "rt") as fh:
        pdb_str = fh.read()
    prot = protein_base.from_pdb_string(
        pdb_str, return_auth_chain_ids=return_auth_chain_ids
    )
    chain_ids = np.unique(prot.chain_index)
    seqs = {}
    for c in chain_ids:
        aatype = prot.aatype[prot.chain_index == c]
        seqs[c.decode()] = protein_base.from_aatype_to_sequence(aatype)
    return prot, seqs

def rename_chains(model_protein, native_chain_map, output_path):
    pdb_str = protein_base.to_pdb(model_protein, native_chain_map)
    with open(output_path, "wt") as fh:
        fh.write(pdb_str)

def fix_model(
    model_path: Path,
    native_path: Path,
    fixed_model_path: Optional[Path] = None,
):
    model_prot, model_seqs = read_pdb_file(model_path)
    native_prot, native_seqs = read_pdb_file(native_path)
    native_chain_order = []
    for c in native_prot.chain_index:
        c = c.decode()
        if c not in native_chain_order:
            native_chain_order.append(c)

    aln_results = aligner.align_chains(model_seqs, native_seqs)

    # Find best assignment, native_chain -> model_chain
    assignment = {}
    for sc, results in aln_results.items():
        sorted_results = sorted(
            results.items(), key=lambda x: x[-1]["aln"].score, reverse=True
        )
        for tc, result in sorted_results:
            if tc in assignment:
                continue
            assignment[tc] = sc
            break

    if len(assignment) != len(model_seqs):
        print(aln_results)
        raise RuntimeError(f"Cannot the model assign automatically {model_path}, {native_path}, {fixed_model_path}")

    # Rename model chains according to the native chain order
    model_prot = dataclasses.asdict(model_prot)
    chain_map = {}
    new_model_prot = defaultdict(list)
    chain_index = model_prot["chain_index"]
    chain_ids = []
    for i, tc in enumerate(native_chain_order):
        if tc not in assignment:
            # print(model_path, native_path, tc, assignment)
            continue
        sc = assignment[tc]
        chain_mask = chain_index == sc.encode()
        model_prot["chain_index"][chain_mask] = i
        chain_map[i] = tc # rename to native chain id
        for k, v in model_prot.items():
            new_model_prot[k].append(v[chain_mask])
        chain_ids.append(tc)

    new_model_prot = {k: np.concatenate(v) for k, v in new_model_prot.items()}
    if fixed_model_path:
        rename_chains(
            protein_base.Protein(**new_model_prot), chain_map, fixed_model_path
        )

    return chain_ids
