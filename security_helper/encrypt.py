import os
import base64
import daiquiri
from Cryptodome import Random
from Cryptodome.Cipher import AES

from . import key as Key


logger = daiquiri.getLogger("security")


def _encrypt(content, key=None):
    if not key:
        key = Key.get_container_key()

    iv = Random.new().read(AES.block_size)
    cipher = AES.new(bytes(key, 'utf-8'), AES.MODE_OFB, iv)
    return base64.b64encode(iv + cipher.encrypt(content))


def encrypt_string(content, key=None):
    if not key:
        key = Key.get_container_key()

    encrypted = _encrypt(content.encode('utf-8'), key)
    return encrypted.decode('utf-8')


def encrypt_file(file_path, ext='', key=None):
    if not key:
        key = Key.get_container_key()

    with open(file_path, 'rb') as _file:
        content = _file.read()

    encrypted = _encrypt(content, key)
    with open(str(file_path) + ext, 'wb') as _encrypted_file:
        _encrypted_file.write(encrypted)


def encrypt_folder(folder_path):
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        if file_name[0] != '.':
            try:
                encrypt_file(f'{folder_path}/{file_name}')
            except Exception as exc:
                logger.exception(exc)
