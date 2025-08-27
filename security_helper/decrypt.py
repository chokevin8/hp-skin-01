import os
import base64
import daiquiri
from Cryptodome.Cipher import AES

from . import key as Key


logger = daiquiri.getLogger("security")


def decrypt(content, key=None):
    try:
        if not key:
            key = Key.get_container_key()

        content = base64.b64decode(content)
        iv = content[:AES.block_size]
        cipher = AES.new(bytes(key, 'utf-8'), AES.MODE_OFB, iv)
        return cipher.decrypt(content[AES.block_size:])
    except Exception as exc:
        logger.error(f'decrypt({content}, {len(key) if key else None}) failed!')
        logger.exception(exc)
        return None


def decrypt_string(content, key=None):
    try:
        if not key:
            key = Key.get_container_key()

        decrypted = decrypt(content.encode('utf-8'), key)
        return decrypted.decode('utf-8')
    except Exception as exc:
        logger.error(f'decrypt_string({content}, {len(key) if key else None}) failed!')
        logger.exception(exc)
        return None


def decrypt_file(file_path, key=None):
    if not key:
        key = Key.get_container_key()

    with open(file_path, 'rb') as _file:
        content = _file.read()

    decrypted = decrypt(content, key)
    with open(file_path, 'wb') as _decrypted_file:
        _decrypted_file.write(decrypted)


def decrypt_folder(file_path):
    # TODO: listdir()의 결과가 폴더면????
    file_names = os.listdir(file_path)
    for file_name in file_names:
        if file_name[0] != '.':
            try:
                decrypt_file(f'{file_path}/{file_name}')
            except Exception as exc:
                logger.exception(exc)
