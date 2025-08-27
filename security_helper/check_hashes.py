import os
import time
import os.path
import daiquiri
import threading
from pathlib import Path
import signal
import platform
import psutil

from . import config as Config
from . import smtp as Smtp
from . import hash as Hash
from . import encrypt as Encrypt
from . import decrypt as Decrypt


logger = daiquiri.getLogger("security")


class CheckHashes(threading.Thread):
    def __init__(self, product_name='', launcher_info=None):
        threading.Thread.__init__(self)

        self.mt_check_count = 0
        self.et_check_count = 0

        self.stop_event = threading.Event()
        self.product_name = product_name
        self.integrity_check_interval = int(os.getenv("INTEGRITY_CHECK_INTERVAL", '300'))
        self.integrity_check_threshold = int(os.getenv("INTEGRITY_CHECK_THRESHOLD", '1'))
        self.launcher_info = launcher_info
        self.last_checked_at = 0    # To prevent from failing before launcher is up at the first time

        logger.debug(f'CheckHashes.__init__(): key_path: {Config.key_path}!')
        logger.debug(f'CheckHashes.__init__(): hashes_path: {Config.hashes_path}!')
        logger.debug(f'CheckHashes.__init__(): app_data_path: {Config.app_data_path}!')
        logger.debug(f'CheckHashes.__init__(): tools_configuration_path: {Config.tools_configuration_path}!')
        logger.debug(f'CheckHashes.__init__(): (op_mode, run_mode, output_mode): ({Config.op_mode}, {Config.run_mode}, {Config.output_mode})!')

    def check_host_launcher(self, one_time=False):
        try:
            status_path = Config.app_data_path / '.launcher-status'
            if status_path.exists():
                retry_mt_done = False
                retry_mt_max = 3
                retry_mt_count = 0
                while not retry_mt_done:
                    # .launcher-status 파일 읽기 (실패 시 재시도 포함)
                    made_time = int(os.path.getmtime(status_path))
                    with open(status_path, 'r', encoding='utf-8') as _status_file:
                        status_content = _status_file.read()
                    if not status_content:
                        retry_mt_count += 1
                        if retry_mt_count < retry_mt_max:
                            logger.warning('Retrying read status_content of launcher status after 1 sec ...')
                            logger.warning(f'The launcher status creation time : {made_time}')
                            time.sleep(1)
                            continue
                        else:
                            if not one_time and self.mt_check_count < self.integrity_check_threshold:
                                logger.info(f'(self.mt_check_count = {self.mt_check_count}) < (self.integrity_check_threshold = {self.integrity_check_threshold})')
                                self.mt_check_count += 1
                                retry_mt_done = True
                            else:
                                logger.error('Cannot read launcher status!')
                                Smtp.set_mail(self.product_name, 'launcher_down')
                                self.stop_event.set()
                                return False

                    # .launcher-status 파일에 저장된 내용과 파일 생성 시간(10초까지 허용)의 hash 값이 다른 경우 처리
                    status_content = int(Decrypt.decrypt_string(status_content))
                    if abs(status_content - made_time) > 10:
                        retry_mt_count += 1
                        if retry_mt_count < retry_mt_max:
                            logger.warning('Retrying check status_content of launcher status after 1 sec ...')
                            logger.warning(f'The launcher status file creation time : {made_time}')
                            logger.warning(f'The launcher status file content : {status_content}')
                            time.sleep(1)
                        else:
                            if not one_time and self.mt_check_count < self.integrity_check_threshold:
                                logger.info(f'(self.mt_check_count = {self.mt_check_count}) < (self.integrity_check_threshold = {self.integrity_check_threshold})')
                                self.mt_check_count += 1
                                retry_mt_done = True
                            else:
                                logger.error('Launcher status is invalid!')
                                Smtp.set_mail(self.product_name, 'launcher_down')
                                self.stop_event.set()
                                return False
                    else:
                        self.mt_check_count = 0
                        retry_mt_done = True

                # one_time인 경우, launcher를 통해 실행되었는지만 체크하는데,
                # last_checked_at이 항상 0이므로 et_done 체크는 무의미하고,
                # 아예 별도로, 생성된지 INTEGRITY_CHECK_INTERVAL 경과 여부만 체크 (.launcher-status 파일 재사용 방지!)
                if one_time:
                    now = int(time.time())
                    if abs(now - status_content) < self.integrity_check_interval:
                        return True
                    else:
                        logger.error('Invalid launcher content!')
                        Smtp.set_mail(self.product_name, 'launcher_down')
                        return False

                retry_et_done = False
                retry_et_max = 3
                retry_et_count = 0
                while not retry_et_done:
                    now = int(time.time())
                    made_time = int(os.path.getmtime(status_path))
                    # 원래 abs(now - made_time) 이었는데,
                    # Windows에서 중간에, 심지어 여러 번 sleep mode에 빠졌다가 깨어난 경우 처리를 위해,
                    # status 파일이 last_checked_at ~ now 사이에 생성된 거면 성공으로 처리!
                    # sleep mode에 빠져 있던 시간을 구해서 제외시키려고 했으나, 여러 번 빠지는 경우 중간에 launcher가 잠깐 살아나면 처리가 안 됨!
                    # 각 프로세스에서 체크 처리에 소요되는 시간 차이 때문에 순서가 바뀔 수가 있어서, 앞에 버퍼 10초 추가함.
                    # 즉, 각 프로세스에서 체크 주기 시작 후 10초 이내에는 처리가 끝나야 함!
                    if self.last_checked_at - 10 <= made_time and made_time <= now:
                        self.et_check_count = 0
                        retry_et_done = True
                    else:
                        retry_et_count += 1
                        if retry_et_count < retry_et_max:
                            logger.warning('Retrying check elapsed_time of launcher status after 1 sec ...')
                            logger.warning(f'Should be (self.last_checked_at - 10 = {self.last_checked_at - 10}) <= (.launcher-status mtime = {made_time}) <= (now = {now})')
                            time.sleep(1)
                        else:
                            if not one_time and self.et_check_count < self.integrity_check_threshold:
                                logger.info(f'(self.et_check_count = {self.et_check_count}) < (self.integrity_check_threshold = {self.integrity_check_threshold})')
                                self.et_check_count += 1
                                retry_et_done = True
                            else:
                                logger.error('Launcher is down!')
                                Smtp.set_mail(self.product_name, 'launcher_down')
                                self.stop_event.set()
                                return False
                self.last_checked_at = now
                logger.debug(f'self.last_checked_at = {self.last_checked_at}')
                return True
            else:
                logger.error('Cannot find launcher status!')
                Smtp.set_mail(self.product_name, 'launcher_down')
                self.stop_event.set()
                return False
        except Exception as exc:
            logger.error('Unknown error while check host launcher!')
            logger.exception(exc)
            Smtp.set_mail(self.product_name, 'unknown_error')
            self.stop_event.set()
            return False

    def get_hash_value(self, target):
        try:
            target_path = ''
            if target == 'app':
                target_path = '/home/deep/' + target
            elif target == '/app' or target == '/base':
                target_path = target
            elif target == 'folder':
                target_path = Path(f'./{self.product_name}')
            with_mtime = Config.run_mode != 'SDK'
            hash_value = Hash.generate_hash_from_file_list_in_folder(target_path, with_mtime)
            return hash_value
        except Exception as exc:
            logger.error(f'Unknown error while get hash value of {target}!')
            logger.exception(exc)
            return None

    def get_hash_from_hash_file(self, target):
        try:
            hashes_path = None
            if target == 'app':
                hashes_path = Config.hashes_path / self.product_name / (target + '.hash')
            elif target == '/app' or target == '/base':
                target = target.replace('/', '')
                hashes_path = Config.hashes_path / self.product_name / (target + '.hash')
            elif target == 'folder':
                hashes_path = Config.hashes_path / (self.product_name + '.hash')

            with open(hashes_path, 'r', encoding='utf-8') as _hash_file:
                file_content = _hash_file.read()

            return file_content.rstrip()
        except Exception as exc:
            logger.error(f'Unknown error while get hash from {hashes_path}!')
            logger.exception(exc)
            if Config.run_mode == 'COMPOSE':
                Smtp.set_mail(self.product_name, 'unknown_error')
                self.stop_event.set()

    def evaluate_hash(self):
        target_list = []
        if Config.run_mode == 'SDK':
            target_list.append('folder')
        else:
            if self.product_name == 'ai-app':
                target_list.append('/app')
                target_list.append('/base')
            else:
                target_list.append('app')

        for target in target_list:
            try:
                if self.get_hash_value(target) != self.get_hash_from_hash_file(target):
                    logger.error(f'Integrity of \'{target}\' is invalid!')
                    if Config.run_mode == 'COMPOSE':
                        Smtp.set_mail(self.product_name, 'invalid_hash')
                        self.stop_event.set()
                        return False
                    else:
                        return False
            except Exception as exc:
                logger.error(f'Unknown error while evaluate {target} hash!')
                logger.exception(exc)
                if Config.run_mode == 'COMPOSE':
                    Smtp.set_mail(self.product_name, 'unknown_error')
                    self.stop_event.set()
                    return False
                else:
                    return False

        logger.debug('Integrity is valid')
        return True

    def make_container_status(self):
        try:
            now = str(int(time.time()))
            container_status = Encrypt.encrypt_string(now)
            with open(Config.app_data_path / f'.{self.product_name}-status', 'w', encoding='utf-8') as _status_file:
                _status_file.write(container_status)
        except Exception as exc:
            logger.error('Unknown error while make container status!')
            logger.exception(exc)
            Smtp.set_mail(self.product_name, 'unknown_error')
            self.stop_event.set()

    def check_hash(self):
        try:
            if Config.run_mode == 'COMPOSE':
                if self.check_host_launcher():
                    if self.evaluate_hash():
                        self.make_container_status()
            else:
                if self.evaluate_hash():
                    return True
        except Exception as exc:
            logger.error('Unknown error while check hash!')
            logger.exception(exc)
            if Config.run_mode == 'COMPOSE':
                Smtp.set_mail(self.product_name, 'unknown_error')
                self.stop_event.set()
            else:
                return False

    def find_process(self, process: dict):
        if process and process['pid']:  # and process['name']:
            for proc in psutil.process_iter():
                try:
                    pinfo = proc.as_dict(attrs=['pid'])
                    if process['pid'] == pinfo['pid']:  # and process['name'].lower() == pinfo['name'].lower():
                        pinfo = proc.as_dict(attrs=['status', 'create_time', 'ppid', 'pid', 'name'])
                        return pinfo
                except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess):
                    pass
        return None

    def check_parent(self):
        if platform.system() == 'Windows':
            pinfo = self.find_process({'pid': self.launcher_info['pid']}) 
            if not pinfo:
                logger.error(f"Cannot find launcher({self.launcher_info['pid']})!")
                self.stop_event.set()

    def run(self):
        if Config.hashes_path:
            while not self.stop_event.is_set():
                start_time = time.time()
                self.check_hash()
                self.check_parent()
                elapsed = time.time() - start_time
                time.sleep(max(0, self.integrity_check_interval - elapsed))  # Adjust for time spent running functions
        else:
            logger.error('Config.hashes_path is not defined!')

        logger.warning('Exiting CheckHashes ...')
        signal.raise_signal(signal.SIGINT)
