import os
import json
import smtplib
import datetime
import daiquiri
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders

from . import config as Config
from . import password as Password


logger = daiquiri.getLogger("security")


def get_configuration():
    if not Config.tools_configuration_path:
        logger.error('Config.tools_configuration_path is not defined!')
        return ''

    config_json_path = Config.tools_configuration_path / 'email.json'
    try:
        with open(config_json_path, 'r', encoding='utf-8') as _json_file:
            loaded_config = json.load(_json_file)

        return loaded_config
    except Exception as exc:
        logger.error(f'Unknown error while get configuration from {config_json_path}!')
        logger.exception(exc)
        return ''


def get_current_time():
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M')
        return nowDatetime


def set_email_info():
    try:
        loaded_config = get_configuration()
        location = loaded_config['location']
        is_email_enabled = loaded_config['enabled']
        smtp_address = loaded_config['smtp_server']['address']
        smtp_port = loaded_config['smtp_server']['port']
        smtp_mtype = loaded_config['smtp_server']['mtype']
        send_from_email = loaded_config['send_from']['id']
        send_from_email_password = loaded_config['send_from']['password']
        send_to_email = loaded_config['send_to']
        return (location, is_email_enabled, smtp_address, smtp_port, smtp_mtype, send_from_email, send_from_email_password, send_to_email)
    except Exception as exc:
        logger.error('Unknown error while set mail info!')
        logger.exception(exc)
        return (None, None, None, None, None, None, None, None)


def set_mail(service_name, status_msg):
    try:
        (location, is_email_enabled, smtp_address, smtp_port, smtp_mtype, send_from_email, send_from_email_password, send_to_email) = set_email_info()
        if is_email_enabled:
            product_name = os.getenv("APP_NAME", "Unknown").upper()
            now_time = get_current_time()
            report = service_name
            header = 'ALERT'
            status = 'Need to be confirmed!'
            font_color = 'red'
            reason = status_msg.capitalize().replace('_', ' ')

            title = f'[{header}] {location} {product_name} Service Status ({now_time})'
            _message = f'[Date/Time] {now_time} <br>'
            _message += f'[Location] {location} <br>'
            _message += f'[Service] {report} <br>'
            _message += f'[Status] <font color="{font_color}">{status}</font> <br>'
            _message += f'[Reason] <font color="{font_color}">{reason}</font> <br>'
            send_mail(
                send_from=send_from_email,
                send_to=send_to_email,
                subject=title,
                message=_message,
                files=[],
                mtype=smtp_mtype,
                server=smtp_address,
                port=smtp_port,
                username=send_from_email,
                password=Password.decrypt_password(send_from_email_password)
            )
    except Exception as exc:
        logger.error('Unknown error while set mail!')
        logger.exception(exc)


def send_mail(send_from, send_to, subject, message, mtype='plain', files=[],
              server="localhost", port=587, username='', password='',
              use_tls=True):
    """Compose and send email with provided info and attachments.

    Args:
        send_from (str): from name
        send_to (list[str]): to name(s)
        subject (str): message title
        message (str): message body
        mtype (str): choose type 'plain' or 'html'
        files (list[str]): list of file paths to be attached to email
        server (str): mail server host name
        port (int): port number
        username (str): server auth username
        password (str): server auth password
        use_tls (bool): use TLS mode
    """
    if password == '':
        logger.error('Wrong password for mail!')
        return

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = ', '.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message, mtype))

    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as _file:
            payload = _file.read()
        part.set_payload(payload)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment', filename=Path(path).name)
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()

    try:
        smtp.login(username, password)
    except Exception as exc:
        logger.error('Unknown error while login mail!')
        logger.exception(exc)
        smtp.quit()

    try:
        smtp.sendmail(send_from, send_to, msg.as_string())
        logger.info('Mail sent successfully!')
    except Exception as exc:
        logger.error('Unknown error while send mail!')
        logger.exception(exc)

    smtp.quit()
