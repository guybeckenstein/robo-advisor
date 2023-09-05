# google drive
import base64
import io
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from service.config import settings
ACCESS_FILE = settings.CONFIG_RESOURCE_LOCATION + 'google_drive_access.json'
MAIN_FOLDER_ID = '1j5GnvPfMrDJpV07fV3YDvoitwBpfZhr6'


def connect_to_google_drive():
    # Path to your credentials JSON file
    credentials_file = ACCESS_FILE
    # Create credentials from the JSON file
    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=['https://www.googleapis.com/auth/drive'])
        # Create a Drive API service
        service = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(e)
        service = None

    return service


class GoogleDriveInstance:
    def __init__(self):
        self.service = connect_to_google_drive()

    def get_files(self):
        # List files and subfolders in the folder
        old_file_id = self.find_file_id_by_name('Top Stocks Israel general bonds indexes intersection.png')
        results = self.service.files().list(q=f"'{MAIN_FOLDER_ID}' in parents").execute()
        files = results.get('files', [])
        if not files:
            print('No files found.')
        else:
            print('Files:')
            for file in files:
                # Download the image
                request = self.service.files().get_media(fileId=file['id'])
                image_data = request.execute()

    def upload_file(self, file_path, num_of_elements):
        folder_id = MAIN_FOLDER_ID
        file_name = file_path.split('/')[-1]

        # Check if a file with the same name exists in the folder
        existing_file_id = self.find_file_id_by_path_name(file_path)

        if existing_file_id:
            self.service.files().delete(fileId=existing_file_id).execute()

        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }

        # Upload the file
        media = MediaFileUpload(file_path, resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media).execute()

    def find_file_id_by_path_name(self, file_path):
        parent_folder_id = MAIN_FOLDER_ID
        folders = file_path.split('/')
        # Initialize the parent folder ID as the root folder ID

        # Iterate through the folder names and find the corresponding folder IDs
        for folder_name in folders[:-1]:
            query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
            folder_results = self.service.files().list(q=query).execute()
            folder_items = folder_results.get('files', [])

            if not folder_items:
                break

            # Update the parent_folder_id with the ID of the found folder
            parent_folder_id = folder_items[0]['id']
        # Check if a file with the same name exists in the folder

        # Search for the file by its name within the parent folder
        file_name = folders[-1]
        query = f"'{parent_folder_id}' in parents and name='{file_name}'"
        file_results = self.service.files().list(q=query).execute()
        files = file_results.get('files', [])

        if not files:
            print(f'File "{file_name}" not found in Google Drive.')
        else:
            return files[0]['id']

    def get_file_by_id(self, file_id):
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
                """request = self.service.files().get_media(fileId=file['id'])
                image_data = request.execute()
                images_list.append(image_data)"""
                images_list.append({'name': file['name'], 'data': self.get_file_by_id(file['id'])})

            return images_list


