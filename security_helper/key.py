import base64
import daiquiri
from Cryptodome import Random
from Cryptodome.Cipher import AES

from . import config as Config
from . import hash as Hash


logger = daiquiri.getLogger("security")


def _get_coordinate(matrix):
    row_list = [6, 12, 8, 13, 5, 14, 10, 15, 2, 0, 11, 3, 9, 4, 1, 7]
    col_list = [14, 3, 2, 5, 15, 0, 7, 12, 8, 1, 4, 11, 6, 13, 10, 9]

    if matrix == 'row':
        return tuple(row_list)
    else:
        return tuple(col_list)


def generate_key_for_key(row_index_tuple=(), col_index_tuple=()):
    if not Config.key_path:
        logger.error('Config.key_path is not defined!')
        return None

    rks_lines = []
    rks_path = Config.key_path / '.rks'
    if not rks_path.exists():
        logger.error(f'{rks_path} does not exist!')
        return None

    try:
        with open(rks_path, 'r', encoding='utf-8') as _file:
            rks_lines = _file.readlines()
        row_index_tuple = _get_coordinate('row') if not row_index_tuple else row_index_tuple
        col_index_tuple = _get_coordinate('col') if not col_index_tuple else col_index_tuple
        key_for_key = ''
        for idx, val in enumerate(row_index_tuple):
            key_for_key += rks_lines[val][col_index_tuple[idx]]

        return bytes(key_for_key, 'utf-8')
    except Exception as exc:
        logger.error(f'Cannot open {rks_path} in generate_key_for_key()!')
        logger.exception(exc)
        return None


def encrypt_key(raw):
    try:
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(generate_key_for_key(), AES.MODE_OFB, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))
    except Exception as exc:
        logger.error('Error occurred on encrypt_key()!')
        logger.exception(exc)
        return None


def _decrypt_key(enc):
    try:
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(generate_key_for_key(), AES.MODE_OFB, iv)
        return cipher.decrypt(enc[AES.block_size:])
    except Exception as exc:
        logger.error('Error occurred on _decrypt_key()!')
        logger.exception(exc)
        return None


def _check_key_files_hash(salt):
    file_list = ['.ck', '.rks']
    for file_name in file_list:
        _file_path = Config.key_path / file_name
        if not Hash.evaluate_hash_file(_file_path, salt):
            logger.error(f'{_file_path} is invalid!')
            return False

    return True


def get_container_key():
    if not Config.key_path:
        logger.error('Config.key_path is not defined!')
        return None

    ck_path = Config.key_path / '.ck'
    if not ck_path.exists():
        logger.error(f'{ck_path} does not exist!')
        return None

    try:
        with open(ck_path, 'rb') as _ck_file:
            content = _ck_file.read()

        decrypted = _decrypt_key(content).decode('utf-8')
        if _check_key_files_hash(decrypted): # salt == decrypted
            return decrypted
        else:
            logger.error("'.ck' or '.rks' is invalid!")
            return None
    except:
        logger.error("Cannot get main key!")
        return None
