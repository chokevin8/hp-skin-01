from pathlib import Path


# For .rks, .ck
key_path: Path = Path('/deep/ai/common')

# For xxx.hash
hashes_path: Path = Path('/deep/ai/hashes')

# For .status
app_data_path: Path = Path('/deep/ai/app/data')

# For email.json
tools_configuration_path: Path = Path('/deep/ai/tools/configuration')

# For op_mode, run_mode, output_mode
op_mode, run_mode, output_mode = 'SECURED', 'COMPOSE', 'FILE'


def set_key_path(folder_path: Path):
    global key_path
    key_path = folder_path


def set_hashes_path(folder_path: Path):
    global hashes_path
    hashes_path = folder_path


def set_app_data_path(folder_path: Path):
    global app_data_path
    app_data_path = folder_path


def set_tools_configuration_path(folder_path: Path):
    global tools_configuration_path
    tools_configuration_path = folder_path


def set_mode(_op_mode: str = 'SECURED', _run_mode: str = 'COMPOSE', _output_mode: str = 'FILE'):
    global op_mode, run_mode, output_mode
    op_mode, run_mode, output_mode = _op_mode, _run_mode, _output_mode
