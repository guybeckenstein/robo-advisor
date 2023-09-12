from service.util import helpers


def save_graph_locally(plt_instance, file_name) -> None:
    helpers.create_graphs_folders_locally()
    # Adjust font size of the table cells
    plt_instance.savefig(f'{file_name}.png', format='png', transparent=True)
    plt_instance.clf()  # Clear the figure after saving
    plt_instance.cla()
    plt_instance.close()


def save_graph_aws_s3(plt_instance, file_name) -> None:
    import boto3
    import io

    from robo_advisor_project import settings

    # Create an S3 client
    s3 = boto3.client('s3')
    # Create an in-memory buffer to store the image data
    buffer = io.BytesIO()

    helpers.create_graphs_folders_locally()
    # Adjust font size of the table cells
    plt_instance.savefig(f'{file_name}.png', format='png', transparent=True)
    plt_instance.clf()  # Clear the figure after saving
    plt_instance.cla()
    plt_instance.close()

    # Upload the image to S3
    buffer.seek(0)  # Reset buffer position to the beginning
    s3.upload_fileobj(buffer, settings.AWS_STORAGE_BUCKET_NAME, f'{file_name}.png')
