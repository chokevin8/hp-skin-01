import base64
import daiquiri
import json
from Cryptodome import Random
from Cryptodome.Cipher import AES
from pathlib import Path
import time

from . import get_mac_addr as Get_Mac_Addr
from . import key as Key
from . import hash as Hash
from . import decrypt as Decrypt


logger = daiquiri.getLogger("license")


def _get_coordinate_for_license(matrix, mac_addr):
    row_list = []
    col_list = []

    # if not mac_addr:
    #     row_list = [14, 3, 2, 5, 15, 0, 7, 12, 8, 1, 4, 11, 6, 13, 10, 9]
    #     col_list = [6, 12, 8, 13, 5, 14, 10, 15, 2, 0, 11, 3, 9, 4, 1, 7]
    # else:
    key = mac_addr.replace(':', '')
    # key = 1a2b3c4d5e6f
    key += key[::-1]
    # key = 1a2b3c4d5e6ff6e5d4c3b2a1

    mac_list = mac_addr.split(':')
    del mac_list[0:2]
    mac_list = list(map(lambda x: x.zfill(2), mac_list))
    # mac_list = [3c, 4d, 5e, 6f]

    first_str = ''
    second_str = ''
    for value in mac_list:
        first_str += value[0]
        second_str += value[1]
    # first_str = 3456
    # second_str = cdef

    key = second_str + first_str + key
    # cdef34561a2b3c4d5e6ff6e5d4c3b2a1

    if len(key) > 32:
        key = key[:32]
    else:
        key = key.zfill(32)

    max = 15
    for cnt in range(max+1):
        hex = int(key[(cnt)*2:(cnt+1)*2], 16)
        row_idx, col_idx = divmod(hex, max)
        if row_idx > max:
            row_idx, col_idx = divmod(row_idx, max)

        row_list.append(row_idx)
        col_list.append(col_idx)

    if matrix == 'row':
        return tuple(row_list)
    else:
        return tuple(col_list)


def encrypt_license(raw, mac_addr: str = None):
    try:
        iv = Random.new().read(AES.block_size)
        if not mac_addr:
            mac_addr = Get_Mac_Addr.get_mac_addr()
            if not mac_addr:
                return None

        cipher = AES.new(Key.generate_key_for_key(_get_coordinate_for_license('row', mac_addr), _get_coordinate_for_license('col', mac_addr)), AES.MODE_OFB, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))
    except Exception as exc:
        logger.error("Cannot encrypt license!")
        logger.exception(exc)
        return None


def _decrypt_license(enc):
    try:
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        mac_addr = Get_Mac_Addr.get_mac_addr()
        if not mac_addr:
            return None

        cipher = AES.new(Key.generate_key_for_key(_get_coordinate_for_license('row', mac_addr), _get_coordinate_for_license('col', mac_addr)), AES.MODE_OFB, iv)
        return cipher.decrypt(enc[AES.block_size:])
    except Exception as exc:
        logger.error("Cannot decrypt license!")
        logger.exception(exc)
        return None


def get_license_key(mac_addr: str = None):
    try:
        if not mac_addr:
            mac_addr = Get_Mac_Addr.get_mac_addr()
            if not mac_addr:
                return None

        license_key = Key.generate_key_for_key(_get_coordinate_for_license('row', mac_addr), _get_coordinate_for_license('col', mac_addr))
        return license_key.decode('utf-8')
    except Exception as exc:
        logger.error("Cannot get license key!")
        logger.exception(exc)
        return None


# model_info 예시) 'DEEP-CHEST-DC-XR-03-V3'
# Error codes:
#   -1: License file not found
#   -2: check_license() exception
#   -3: License key not found
#   -4: No license key
#   -5: Invalid activated_at
#   -6: Model not found in license
#   -7: Invalid license_expired_time
#   -8: The license has expired
def check_license(model_info: str, api_key: str = None):
    license_path = '/deep/ai/common/license.ck'
    license_info = None
    if Path(license_path).exists():
        if Hash.evaluate_hash_file(license_path):
            try:
                with open(license_path, 'rb') as file:
                    content = file.read()
                decrypted = Decrypt.decrypt(content)
                if decrypted:
                    license_json = json.loads(decrypted)
                    if api_key:
                        license_info = license_json[api_key] if api_key in license_json else None
                        if not license_info:
                            return (False, 'The license is invalid (-3)')
                    else:
                        # TODO: DICOM 통신 방식에서는 어느 key를 써야할지 알 수 없기 때문에 무조건 첫번째 key 사용!
                        license_keys = list(license_json.keys())
                        license_info = license_json[license_keys[0]] if license_keys else None
                        if not license_info:
                            return (False, 'The license is invalid (-4)')
                else:
                    return (False, 'Unknown error while decrypt license file')
            except Exception as exc:
                return (False, f'Unknown error while check license file\n{exc}')
    else:
        return (False, 'The license is invalid (-1)')

    try:
        activated_at = license_info.get('activated_at', '')
        if not isinstance(activated_at, int) and len(activated_at) == 0:
            return (False, 'The license is invalid (-5)')

        if model_info not in license_info["model_list"]:
            return (False, 'The license is invalid (-6)')

        license_expired_time = license_info["model_list"][model_info].get('expiration_time', '')
        if not isinstance(license_expired_time, int) and len(license_expired_time) == 0:
            return (False, 'The license is invalid (-7)')

        if license_expired_time != 0:
            now = int(time.time())
            if (now > license_expired_time) and (activated_at != license_expired_time):
                return (False, 'The license is invalid (-8)')

        # TODO: 판독 건수 체크
        # max_inference_count = license_info.get('max_inference_count', '')
        # if not isinstance(max_inference_count, int) and len(max_inference_count) == 0:
        #     return (False, 'Invalid max_inference_count')

        # if max_inference_count != 0:
        #     done_inference_count = license_info.get('done_inference_count', '')
        #     if not isinstance(done_inference_count, int) and len(done_inference_count) == 0:
        #         return (False, 'Invalid done_inference_count')

        #     if max_inference_count < done_inference_count:
        #         return (False, 'Maximum number of inferencing exceeded!')

        return (True, '')
    except Exception as exc:
        return (False, f'The license is invalid (-2)\n{exc}')
