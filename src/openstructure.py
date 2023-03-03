import re
import logging
import dataclasses
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from collections import defaultdict, OrderedDict

import numpy as np

logger = logging.getLogger(__file__)


@dataclasses.dataclass(frozen=True)
class OpenStructureResult:
    lddt: float
    qs: float
    msg: str


THIRD_PARTY_ROOT = Path(__file__).resolve().parents[2].joinpath("third_party")

class OpenStructureRunner:
    def __init__(self, image_name: str):
        super().__init__()

        self.image_name = image_name

    def run_batch(
        self,
        model_files: List[Path],
        native: Path,
        output_files: List[Path],
        output_dir: Path,
        use_docker: bool = True,
    ):
        if use_docker:
            with tempfile.TemporaryDirectory(prefix="ost_docker_batch", dir="/tmp") as tmp_dir:
                self._run_docker(model_files, native, output_files, output_dir, Path(tmp_dir))

    def run(
        self, model: Path, native: Path, output_file: Path, use_docker: bool = True
    ):
        if use_docker:
            with tempfile.TemporaryDirectory(prefix="ost_docker", dir="/tmp") as tmp_dir:
                self._run_docker_single(model, native, output_file, Path(tmp_dir))

    def _run_docker(
        self,
        model_files: List[Path],
        native_file: Path,
        output_files: List[Path],
        output_dir: Path,
        tmp_dir: Path,
    ):
        # Prepare working directory
        script_file = THIRD_PARTY_ROOT.joinpath("run_openstructure_docker.py")
        tmp_script_file = tmp_dir.joinpath(script_file.name)
        shutil.copyfile(script_file, tmp_script_file)

        tmp_ref_dir = tmp_dir.joinpath("refs")
        tmp_ref_file = tmp_ref_dir.joinpath(native_file.name)
        tmp_ref_dir.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(native_file, tmp_ref_file)

        tmp_input_dir = tmp_dir.joinpath("inputs")
        tmp_input_dir.mkdir(exist_ok=True, parents=True)

        tmp_output_dir = tmp_dir.joinpath("outs")
        tmp_output_dir.mkdir(exist_ok=True, parents=True)

        tmp_list_file = tmp_dir.joinpath("inputs.txt")
        output_file_names = set()
        with open(tmp_list_file, "wt") as fh:
            for model_file, output_file in zip(model_files, output_files):
                output_file_name = output_file.name
                assert output_file_name not in output_file_names
                output_file_names.add(output_file_name)
                fh.write(f"{model_file.name} {native_file.name} {output_file_name}\n")
                tmp_model_file = tmp_input_dir.joinpath(model_file.name)
                shutil.copyfile(model_file, tmp_model_file)

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(tmp_dir)}:/home",
            self.image_name,
            tmp_script_file.name,
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        output_str = output.decode()
        retcode = process.wait()

        if retcode:
            print(cmd)
            print(stderr)
            print(output_str)
            raise RuntimeError("Error during batch running Openstructure Docker")

        shutil.copytree(tmp_output_dir, output_dir, dirs_exist_ok=True)

    def _run_docker_single(self, model: Path, native: Path, output_file: Path, tmp_dir: Path):

        tmp_model = tmp_dir.joinpath("model.pdb")
        shutil.copyfile(model, tmp_model)
        tmp_native = tmp_dir.joinpath("ref.pdb")
        shutil.copyfile(native, tmp_native)

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(tmp_dir)}:/home",
            self.image_name,
            "compare-structures",
            "--model", str(tmp_model.name),
            "--reference", str(tmp_native.name),
            "--output", "output.json",
            "--qs-score",
            "--residue-number-alignment",
            "--lddt",
            # "--structural-checks",
            "--consistency-checks",
            "--inclusion-radius", "15.0",
            "--bond-tolerance", "15.0",
            "--angle-tolerance", "15.0",
            "--molck",
            "--remove", "oxt hyd unk",
            "--clean-element-column",
            "--map-nonstandard-residues",
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        output_str = output.decode()
        retcode = process.wait()

        if retcode:
            print(cmd)
            print(stderr)
            print(output_str)
            raise RuntimeError("Error during running Openstructure Docker")

        tmp_output = tmp_dir.joinpath("output.json")
        assert tmp_output.exists(), tmp_output
        shutil.copyfile(tmp_output, output_file)
