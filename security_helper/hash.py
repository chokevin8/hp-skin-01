import os
import hashlib
import daiquiri
from pathlib import Path

from . import key as Key
from . import config as Config


logger = daiquiri.getLogger("security")


def generate_hash(_content, rounds=1, salt=''):
    result = _content
    if salt == '':
        salt = Key.get_container_key()

    for _ in range(rounds):
        result = hashlib.sha512((str(result) + salt).encode("utf-8")).hexdigest().upper()
    return result


def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if '__pycache__' in entry.path:
            continue

        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry


def generate_hash_from_file_list_in_folder(folder, with_mtime):
    merged_list = []

    entries = scantree(folder)
    for entry in entries:
        entry_info = ''
        entry_info += str(entry.name)
        entry_info += str(entry.stat().st_size)

        # SDK에서는 unzip 시, 재 배포 시 등 날짜가 바뀌는 상황이 생길 수 밖에 없음!
        # 사실 대부분 binary나, encrypted 파일이라 위변조 시 오류가 발생할 듯...
        # 그래서 위변조 체크는 되지 않겠지만, 날짜도 제외함!
        # _make_folder_hash.py에서 docker cp 시 st_mtime의 소수점 이하는 유실됨!
        if with_mtime:
            entry_info += str(int(entry.stat().st_mtime))
        merged_list.append(entry_info)

    entries.close()

    # locale 등 설정에 따라 파일 목록 결과 순서가 다를 수 있음!
    merged_info = ''.join(sorted(merged_list))

    result = generate_hash(merged_info)
    return result


# TODO: 속도 때문에 위에 있는 generate_hash_from_file_list_in_folder()로 바꿈!
def generate_hash_from_file_contents_in_folder(folder):
    hash_list = []
    for (root, _, files) in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            hash_list.append(generate_hash_from_file(file_path))

    sum_hash = ''
    for _hash in sorted(hash_list):
        sum_hash += _hash

    result = generate_hash(sum_hash)
    return result


def generate_hash_from_file(filename, salt=''):
    with open(filename, 'rb') as _file:
        file_content = _file.read()

    result = generate_hash(file_content, salt=salt)
    return result


def create_hash_file(target, type='file'):
    try:
        with_mtime = False if Config.run_mode == 'SDK' else True
        if type == 'file':
            result = generate_hash_from_file(target)
        # SDK
        elif type == 'file_list_in_folder':
            result = generate_hash_from_file_list_in_folder(target, with_mtime)
        else:
            # result = generate_hash_from_file_contents_in_folder(target)
            result = generate_hash_from_file_list_in_folder(target, with_mtime)

        hash_path = Path(os.path.dirname(target)) / f'{os.path.basename(target)}.hash'
        with open(hash_path, 'w', encoding='utf-8') as _hash_file:
            _hash_file.write(result)

        return True
    except:
        return False


def print_log_once(msg, log_history: list):
    if msg not in log_history:
        log_history.append(msg)
        logger.info(msg)


def evaluate_hash_file(_file, salt=''):
    if (Config.op_mode == 'DEV') or (Config.op_mode == 'CLOSED'):
        return True

    # TODO: 성공 로그 남기기?
    # log_history = []

    # On WSL2, there was an error to evaluate_hash_file()!
    # To recover from it, retry 3 times!
    retry_count = 3
    while retry_count > 0:
        _file = Path(str(_file))
        hash_path = Path(str(_file) + '.hash')
        if not hash_path.exists():
            retry_count -= 1
            if retry_count == 0:
                logger.error(f'{hash_path} does not exist!')
            continue

        with open(hash_path, 'r', encoding='utf-8') as _hash_file:
            stored_hash = _hash_file.read()

        if (len(stored_hash) == 0):
            retry_count -= 1
            if retry_count == 0:
                logger.error(f'{stored_hash} is empty!')
            continue

        try:
            if generate_hash_from_file(_file, salt=salt) == stored_hash:
                # TODO: 성공 로그 남기기?
                # file_name = _file.name
                # if file_name != '.ck' and file_name != '.rks':
                #     print_log_once(f'{file_name} is valid', log_history)
                return True
            else:
                retry_count -= 1
                if retry_count == 0:
                    logger.error(f'Hash from {_file} is not same as {hash_path}!')
                continue
        except:
            retry_count -= 1
            if retry_count == 0:
                logger.error(f'Cannot generate hash from {_file}!')
            continue

    return False
