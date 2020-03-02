# -*- coding: utf-8 -*-
"""
Spyder Editor

from this StackOverflow answer: https://stackoverflow.com/a/39225039
"""
import requests
import zipfile
import warnings
from sys import stdout
from os import makedirs, remove
from os.path import dirname, exists
from glob import glob


def debug_print_files():
    all_files = glob('./**/*')
    print(all_files)

class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, unzip=False, showsize=False, del_zip=False):
        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()

            debug_print_files()
            print('Downloading {} into {}... '.format(
                file_id, dest_path), end='')
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={
                                   'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(
                    GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            if showsize:
                print()  # Skip to the next line

            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(
                response, dest_path, showsize, current_download_size)
            print('Done.')
            debug_print_files()

            if unzip:
                try:
                    print('Unzipping...', end='')
                    stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    debug_print_files()
                    print('Done.')
                    if del_zip:
                        try:
                            remove(dest_path)
                            print('Deleted zip file.')
                        except Exception as error:
                            warnings.warn(
                                'Could not delete zip file, error: {}'.format(str(error)))
                except zipfile.BadZipfile:
                    warnings.warn(
                        'Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        print(
                            '\r' + GoogleDriveDownloader.sizeof_fmt(current_size[0]), end=' ')
                        stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}'.format(num, unit, suffix)
            num /= 1024.0
        return '{:.1f} {}{}'.format(num, 'Yi', suffix)
