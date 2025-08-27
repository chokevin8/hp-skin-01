import os
import sys
from os import listdir
from os.path import islink, realpath, join
from pathlib import Path
import platform
if platform.system() == 'Linux':
    import netifaces
import subprocess
import daiquiri
import time
import shutil


logger = daiquiri.getLogger("security")
mac_addr = None


# Priority:
#   Fixed Ethernet > Fixed Wi-Fi > USB Ethernet (only if ALLOW_USB_NIC == true) > USB Wi-Fi (only if ALLOW_USB_NIC == true)
def get_mac_addr():
    global mac_addr
    if mac_addr:
        return mac_addr

    if platform.system() == 'Linux':
        is_wsl2 = True if ('microsoft' in platform.uname().release) and ('WSL2' in platform.uname().release) else False
        if is_wsl2:
            windows_user = None
            success = False
            retry_count = 0
            while not success and retry_count < 3:
                windows_user = subprocess.check_output('wslvar USERNAME'.split(), text=True).strip()
                if windows_user:
                    success = True
                else:
                    logger.warning('Retrying to get Windows Username ...')
                    retry_count = retry_count + 1
                    time.sleep(1)
            if not success:
                logger.error('Failed to get Windows Username!')
                return False

            wudid_filename = '.wudid'
            exe_path = Path(f'/mnt/c/Users/{windows_user}/.deep-ai/deep-ai.exe')
            content = ''
            if exe_path.exists():
                if getattr(sys, 'frozen', False):   # binary by PyInstaller
                    tools_folder = os.path.dirname(os.path.realpath(sys.executable))
                else:                               # .py using python
                    tools_folder = os.path.dirname(os.path.realpath(__file__))
                wudid_path = Path(f'{tools_folder}/{wudid_filename}')
                command = f'{exe_path} {tools_folder}'
                os.system(command)

                if wudid_path.exists():
                    with open(wudid_path, 'r', encoding='utf-8') as _wudid_file:
                        content = _wudid_file.read()
                    if content:
                        from . import key as Key
                        content = Key._decrypt_key(content.encode('utf-8')).decode('utf-8')
                    else:
                        logger.error(f'No contents in {wudid_path}!')
                    os.remove(wudid_path)
                else:
                    logger.error(f'{wudid_path} does not exist!')
            else:
                logger.error(f'{exe_path} does not exist!')

            if content:
                mac_addr = content
                logger.debug(f'MAC from deep-ai.exe: {mac_addr}')
                return mac_addr
            else:
                logger.error('Failed to get info from deep-ai.exe!')
                return False
        else:
            interfaces = [dir for dir in listdir('/sys/class/net') if islink(join('/sys/class/net', dir))]
            interfaces = [dir for dir in interfaces if not realpath(join('/sys/class/net', dir)).startswith('/sys/devices/virtual')]
            interfaces = [dir for dir in interfaces if not realpath(join('/sys/class/net', dir)).startswith('/sys/devices/usb')]
            # Ethernet would be start with "e" whereas Wi-Fi with "w",
            # so, Ethernet adapter would be checked at first by sorting.
            interfaces.sort()
            if len(interfaces) >= 1:
                mac_addr = netifaces.ifaddresses(interfaces[0])[netifaces.AF_PACKET][0]['addr']
                logger.debug(f'MAC from netifaces: {mac_addr}')
                return mac_addr
            elif os.getenv('ALLOW_USB_NIC', 'False').lower() == 'true':
                interfaces = [dir for dir in listdir('/sys/class/net') if islink(join('/sys/class/net', dir))]
                interfaces = [dir for dir in interfaces if not realpath(join('/sys/class/net', dir)).startswith('/sys/devices/virtual')]
                # Ethernet would be start with "e" whereas Wi-Fi with "w",
                # so, Ethernet adapter would be checked at first by sorting.
                interfaces.sort()
                if len(interfaces) >= 1:
                    mac_addr = netifaces.ifaddresses(interfaces[0])[netifaces.AF_PACKET][0]['addr']
                    logger.debug(f'MAC from netifaces: {mac_addr}')
                    return mac_addr
                else:
                    logger.error('Failed to get info from netifaces!')
                    return False
            else:
                logger.error('Failed to get info from netifaces!')
                return False
    elif platform.system() == 'Windows':
        # getmac is not working if not connected!

        # 관련 PowerShell 명령어 참고)
        # Get-NetAdapter -Physical -IncludeHidden * | Format-List -Property *
        # Get-NetAdapter -Physical -IncludeHidden * | Format-List -Property "Name", "InterfaceDescription", "PhysicalMediaType"
        # powershell -Command 'Get-NetAdapter -Physical -IncludeHidden * | Format-List -Property \"Name\", \"InterfaceDescription\", \"PhysicalMediaType\"'
        # powershell -Command 'Get-NetAdapter -Physical -IncludeHidden | Where { ($_.PhysicalMediaType -eq \"802.3\" -or $_.PhysicalMediaType -eq \"Native 802.11\") } | Sort Name | Format-Table MacAddress'

        powershell_exe = 'powershell.exe'
        if not shutil.which(powershell_exe):
            logger.warning(f'Cannot find {powershell_exe} in the PATH!')
            powershell_exe = '%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\' + powershell_exe
            powershell_exe = os.path.expandvars(powershell_exe)
            if not (os.path.exists(powershell_exe) and os.path.isfile(powershell_exe) and os.access(powershell_exe, os.X_OK)):
                logger.error(f'Failed to get {powershell_exe}!')
                return False

        # For the first time, try to get MAC address from Ethernet adapter not USB type
        cmd = 'Get-NetAdapter -Physical -IncludeHidden | Where { ($_.PhysicalMediaType -eq "802.3" -and -not $_.ComponentID.StartsWith("USB")) } | Sort Name | Format-Table MacAddress'
        result = subprocess.check_output([powershell_exe, '-Command', cmd], creationflags=subprocess.CREATE_NO_WINDOW, text=True).strip()
        result = result.splitlines(keepends=True)
        if len(result) >= 3:
            mac_addr = ''.join(result[2]).rstrip().lower().replace('-', ':')
            logger.debug(f'MAC from Get-NetAdapter (802.3): {mac_addr}')
            return mac_addr
        else:
            # If failed, try to get MAC address from Wi-Fi adapter not USB type
            cmd = 'Get-NetAdapter -Physical -IncludeHidden | Where { ($_.PhysicalMediaType -eq "Native 802.11" -and -not $_.ComponentID.StartsWith("USB")) } | Sort Name | Format-Table MacAddress'
            result = subprocess.check_output([powershell_exe, '-Command', cmd], creationflags=subprocess.CREATE_NO_WINDOW, text=True).strip()
            result = result.splitlines(keepends=True)
            if len(result) >= 3:
                mac_addr = ''.join(result[2]).rstrip().lower().replace('-', ':')
                logger.debug(f'MAC from Get-NetAdapter (Native 802.11): {mac_addr}')
                return mac_addr
            elif os.getenv('ALLOW_USB_NIC', 'False').lower() == 'true':
                # Including Ethernet adapter of USB type
                cmd = 'Get-NetAdapter -Physical -IncludeHidden | Where { ($_.PhysicalMediaType -eq "802.3") } | Sort Name | Format-Table MacAddress'
                result = subprocess.check_output([powershell_exe, '-Command', cmd], creationflags=subprocess.CREATE_NO_WINDOW, text=True).strip()
                result = result.splitlines(keepends=True)
                if len(result) >= 3:
                    mac_addr = ''.join(result[2]).rstrip().lower().replace('-', ':')
                    logger.debug(f'MAC from Get-NetAdapter (802.3 USB): {mac_addr}')
                    return mac_addr
                else:
                    # Including Wi-Fi adapter of USB type
                    cmd = 'Get-NetAdapter -Physical -IncludeHidden | Where { ($_.PhysicalMediaType -eq "Native 802.11") } | Sort Name | Format-Table MacAddress'
                    result = subprocess.check_output([powershell_exe, '-Command', cmd], creationflags=subprocess.CREATE_NO_WINDOW, text=True).strip()
                    result = result.splitlines(keepends=True)
                    if len(result) >= 3:
                        mac_addr = ''.join(result[2]).rstrip().lower().replace('-', ':')
                        logger.debug(f'MAC from Get-NetAdapter (Native 802.11 USB): {mac_addr}')
                        return mac_addr
                    else:
                        logger.error('Failed to get info from Get-NetAdapter!')
                        return False
            else:
                logger.error('Failed to get info from Get-NetAdapter!')
                return False
    else:
        logger.error('Not supported platform!')
        return False
