import base64
from Cryptodome import Random
from Cryptodome.Cipher import AES

from . import key as Key


def get_hash_key():
    return bytes(Key.get_container_key(), 'utf-8')


def _encrypt(raw):
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(get_hash_key(), AES.MODE_OFB, iv)
    return base64.b64encode(iv + cipher.encrypt(raw.encode('utf-8')))


def encrypt_password(password):
    return _encrypt(password).decode('utf-8')


def _decrypt(enc):
    enc = base64.b64decode(enc)
    iv = enc[:AES.block_size]
    cipher = AES.new(get_hash_key(), AES.MODE_OFB, iv)
    return cipher.decrypt(enc[AES.block_size:])


def decrypt_password(enc):
    if type(enc) == str:
        enc = str.encode(enc)

    return _decrypt(enc).decode('utf-8')
