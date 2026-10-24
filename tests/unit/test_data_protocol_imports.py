"""Protocol/evidence inspection remains usable without loading the ML stack."""

import subprocess
import sys


def test_protocol_cold_import_does_not_load_training_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from src.data.sft_protocol import protocol_identity; "
                "assert protocol_identity(None)['protocol'] == 'alpaca_full_prompt_v1'; "
                "assert not {'torch', 'transformers', 'datasets'} & sys.modules.keys()"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_public_dataset_exports_still_resolve_to_existing_classes():
    from src.data import AlpacaDataset, BaseDataset, MedicalEntityDataset
    from src.data.base import BaseDataset as ActualBase
    from src.data.loaders import AlpacaDataset as ActualAlpaca
    from src.data.medical_dataset import MedicalEntityDataset as ActualMedical

    assert (AlpacaDataset, BaseDataset, MedicalEntityDataset) == (
        ActualAlpaca,
        ActualBase,
        ActualMedical,
    )
