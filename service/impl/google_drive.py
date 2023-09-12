# google drive
import os
import io

from google.oauth2 import service_account

import googleapiclient
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from service.config import settings

# Global variables
ACCESS_FILE: str = f'{settings.CONFIG_RESOURCE_LOCATION}google_drive_access_new.json'
MAIN_FOLDER_ID: str = '1yTCkRBihTdF-mTdKhZH_GrF3Ir7bHoL7'


def connect_to_google_drive():
    # Path to your credentials JSON file
    credentials_file = ACCESS_FILE
    # Create credentials from the JSON file
    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=['https://www.googleapis.com/auth/drive'])
        # Create a Drive API service
        service = build(serviceName='drive', version='v3', credentials=creds)
    except Exception as e:
        print(e)
        service = None

    return service


class GoogleDriveInstance:
    def __init__(self):
        self.service = connect_to_google_drive()

    def get_files(self):
        # List files and subfolders in the folder
        results = self.service.files().list(q=f"'{MAIN_FOLDER_ID}' in parents").execute()
        files = results.get('files', [])
        if not files:
            print('No files found.')
        else:
            print('Files:')
            for file in files:
                # Download the image
                request = self.service.files().get_media(fileId=file['id'])
                request.execute()

    def upload_file(self, fully_qualified_file_path: str, shortened_file_path: str):
        folder_id: str = self.find_file_id_by_path_name(shortened_file_path, folder_type=True)
        file_name = shortened_file_path.split('/')[-1]
        media: googleapiclient.http.MediaFileUpload = MediaFileUpload(fully_qualified_file_path, resumable=True)
        file_metadata: dict[str] = {
            'name': file_name,
        }

        # Check if a file with the same name exists in the folder
        existing_file_id: str = self.find_file_id_by_path_name(shortened_file_path)

        if existing_file_id:
            # Add the new parent folder
            self.service.files().update(
                body=file_metadata, media_body=media, fileId=existing_file_id, addParents=folder_id
            ).execute()
        else:
            # Create the file and specify the parent folder using addParents
            file_metadata['parents'] = [folder_id]
            self.service.files().create(
                body=file_metadata, media_body=media, fields='id', supportsAllDrives=True
            ).execute()

    def find_file_id_by_path_name(self, file_path, folder_type=False):
        parent_folder_id = MAIN_FOLDER_ID
        folders = file_path.split('/')
        # Iterate through the folder names and find the corresponding folder IDs
        for folder_name in folders[:-1]:
            query = (f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' "
                     f"and name='{folder_name}'")
            folder_results = self.service.files().list(q=query).execute()
            folder_items = folder_results.get('files', [])

            if not folder_items:
                break

            # Update the parent_folder_id with the ID of the found folder
            parent_folder_id = folder_items[0]['id']
        # Check if a file with the same name exists in the folder

        if folder_type:
            return parent_folder_id

        # Search for the file by its name within the parent folder
        file_name = folders[-1]
        query = f"'{parent_folder_id}' in parents and name='{file_name}'"
        file_results = self.service.files().list(q=query).execute()
        files = file_results.get('files', [])

        if not files:
            print(f'File "{file_name}" not found in Google Drive.')
        else:
            return files[0]['id']

    def get_file_by_id(self, file_id: str):
        request = self.service.files().get_media(fileId=file_id)
        # Create an in-memory binary stream to store the downloaded content
        file_stream = io.BytesIO()

        # Create a MediaIoBaseDownload object to download the file content in binary format
        downloader = MediaIoBaseDownload(file_stream, request)

        # Download the file content
        done = False
        while not done:
            status, done = downloader.next_chunk()

        # After the download is complete, 'file_stream' contains the binary content of the file

        # Reset the file stream position to the beginning
        file_stream.seek(0)

        return file_stream

    def get_file_by_path(self, file_path):
        file_id = self.find_file_id_by_path_name(file_path)
        return self.get_file_by_id(file_id)

    def get_all_png_files(self):
        folder_id = self.find_file_id_by_path_name(F'{MAIN_FOLDER_ID}/research')
        query = f"'{folder_id}' in parents and mimeType='image/png'"
        results = self.service.files().list(q=query).execute()
        files = results.get('files', [])
        images_list = []

        if not files:
            print('No PNG files found.')
        else:
            for file in files:
                images_list.append({'name': file['name'], 'data': self.get_file_by_id(file['id'])})

            return images_list

    def get_csv_files(self, parent_dir):
        parent_id = self.find_file_id_by_path_name(os.path.dirname(parent_dir))
        query = f"'{parent_id}' in parents and mimeType='text/csv'"
        results = self.service.files().list(q=query).execute()
        files = results.get('files', [])
        csv_files_list = []

        if not files:
            print('No CSV files found.')
        else:
            for file in files:
                csv_files_list.append({'name': file['name'], 'data': self.get_file_by_id(file['id'])})

            return csv_files_list
