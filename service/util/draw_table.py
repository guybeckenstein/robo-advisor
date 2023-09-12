import textwrap
import platform
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

from service.util import helpers

# Global constants
CELL_WIDTH: list[int, int, int] = [200, 105, 1255]
CELL_HEIGHT = 40
TABLE_PADDING = 10
if platform.system() == 'Windows':
    FONTS: list[str, str, str] = [
        'arialbd.ttf',
        'ariali.ttf',
        'arial.ttf',
    ]
else:
    FONTS: list[str, str, str] = [
        '/usr/src/app/static/fonts/arialbd.ttf',
        '/usr/src/app/static/fonts/ariali.ttf',
        '/usr/src/app/static/fonts/arial.ttf',
    ]


def draw_all_and_save_as_png_locally(file_name: str, symbols: list[str], values: list[float], descriptions: list[str],
                                     header_text, percent_mode: bool = True) -> None:
    """
    Stocks Representation
    """
    if len(values) != len(symbols) or len(values) != len(descriptions):
        raise ValueError("Input lists must have the same length.")
    image, draw = _create_blank_image(num_rows=len(symbols))
    _draw_table_header(draw=draw, header_text=header_text)
    _draw_table_rows(symbols=symbols, values=values, descriptions=descriptions, draw=draw, percent_mode=percent_mode)
    image.save(f"{file_name}.png")  # Save the image


def _create_blank_image(num_rows: int) -> tuple[Image, ImageDraw]:
    # Calculate image size
    image_width = CELL_WIDTH[0] + CELL_WIDTH[1] + CELL_WIDTH[2] + 2 * TABLE_PADDING
    image_height = (num_rows + 1) * CELL_HEIGHT + 2 * TABLE_PADDING

    # Create a blank image
    image = Image.new(mode="RGB", size=(image_width, image_height), color="white")
    draw = ImageDraw.Draw(image)
    return image, draw


def _draw_table_header(draw: ImageDraw, header_text: ImageFont.truetype) -> None:
    header_font = ImageFont.truetype(font=FONTS[0], size=26)
    # header_text: list[str, str, str] = ['Stock', 'Weight', 'Description']
    for col_idx in range(len(CELL_WIDTH)):
        x0: int = _calculate_cumulative_col_offset(col_idx) + TABLE_PADDING
        y0: int = TABLE_PADDING
        x1: int = x0 + CELL_WIDTH[col_idx]
        y1: int = y0 + CELL_HEIGHT
        draw.rectangle(xy=[(x0, y0), (x1, y1)], fill="lightgray", outline="black")
        draw.text(xy=(x0 + 5, y0 + 5), text=header_text[col_idx], fill="black", font=header_font)


def _draw_table_rows(symbols: list[str], values: list[float], descriptions: list[str],
                     draw: ImageDraw, percent_mode: bool = False) -> None:
    x0: list[int, int, int] = [
        _calculate_cumulative_col_offset(0) + TABLE_PADDING,
        _calculate_cumulative_col_offset(1) + TABLE_PADDING,
        _calculate_cumulative_col_offset(2) + TABLE_PADDING,
    ]
    x1: list[int, int, int] = [x0[0] + CELL_WIDTH[0], x0[1] + CELL_WIDTH[1], x0[2] + CELL_WIDTH[2]]
    row_font = ImageFont.truetype(font=FONTS[1], size=26)
    for row_idx, (symbol, weight, description) in enumerate(zip(symbols, values, descriptions)):
        y0: int = (row_idx + 1) * CELL_HEIGHT + TABLE_PADDING
        y1: int = y0 + CELL_HEIGHT
        draw.rectangle(xy=[(x0[0], y0), (x1[0], y1)], fill="white", outline="black")
        draw.text(xy=(x0[0] + 5, y0 + 5), text=str(symbol), fill="black", font=row_font)
        draw.rectangle(xy=[(x0[1], y0), (x1[1], y1)], fill="white", outline="black")
        if percent_mode:
            draw.text(xy=(x0[1] + 5, y0 + 5), text=f"{weight:.1%}", fill="black", font=row_font)
        else:
            draw.text(xy=(x0[1] + 5, y0 + 5), text=f"{weight:.0f}", fill="black", font=row_font)
        draw.rectangle(xy=[(x0[2], y0), (x1[2], y1)], fill="white", outline="black")
        draw.text(xy=(x0[2] + 5, y0 + 5), text=str(description), fill="black", font=row_font)


def _calculate_cumulative_col_offset(col_idx: int) -> int:
    res_sum: int = 0
    for i in range(col_idx):
        res_sum += CELL_WIDTH[i]
    return res_sum


def draw_all_and_save_as_png_aws_s3(file_name: str, symbols: list[str], values: list[float], descriptions: list[str],
                                    header_text, percent_mode: bool = True) -> None:
    import io
    import boto3
    from robo_advisor_project.settings import AWS_STORAGE_BUCKET_NAME

    """
    Stocks Representation
    """
    if len(values) != len(symbols) or len(values) != len(descriptions):
        raise ValueError("Input lists must have the same length.")
    image, draw = _create_blank_image(num_rows=len(symbols))
    _draw_table_header(draw=draw, header_text=header_text)
    _draw_table_rows(symbols=symbols, values=values, descriptions=descriptions, draw=draw, percent_mode=percent_mode)

    # Save the image to an in-memory buffer
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    # Upload the image buffer to S3
    s3 = boto3.client('s3')
    bucket_name = AWS_STORAGE_BUCKET_NAME
    s3_key = f"{file_name}.png"  # S3 object key

    try:
        s3.upload_fileobj(image_buffer, bucket_name, s3_key)
    except Exception as e:
        print(f"Error uploading image to S3: {e}")


def draw_research_table(path, table_data, labels, sort_by_option: bool,
                        ending_file_name: str, colors: list[str]) -> None:
    if sort_by_option:
        table_data = table_data.sort_values(by=labels[0], ascending=False).head(10)
    num_of_stocks_showing = 10
    descriptions = helpers.get_stocks_descriptions(table_data.index.values)[1:]

    # Add "Stock" column to intersection_data.columns
    column_headers = ['Stock'] + labels

    # Define cell dimensions and font size
    CELL_WIDTH = 170
    CELL_HEIGHT = 90
    TABLE_PADDING = 10
    # Define font sizes for titles and values
    TITLE_FONT_SIZE = 20
    VALUE_FONT_SIZE = 22
    # Load title and value fonts with the specified sizes
    title_font = ImageFont.truetype(font=FONTS[2], size=TITLE_FONT_SIZE)
    value_font = ImageFont.truetype(font=FONTS[2], size=VALUE_FONT_SIZE)

    # Calculate image dimensions based on the number of rows and columns
    num_rows = min(num_of_stocks_showing, len(table_data)) + 1
    num_cols = len(table_data.columns) + 1
    image_width = CELL_WIDTH * num_cols + TABLE_PADDING
    image_height = CELL_HEIGHT * num_rows + TABLE_PADDING

    # Create a blank image
    image = Image.new(mode="RGB", size=(image_width, image_height), color="white")
    draw = ImageDraw.Draw(image)

    # Draw headers
    headers = ['Stock'] + list(table_data.columns)
    for col_idx, header in enumerate(headers):
        x0 = col_idx * CELL_WIDTH + TABLE_PADDING
        y0 = 0
        x1 = (col_idx + 1) * CELL_WIDTH + TABLE_PADDING
        y1 = CELL_HEIGHT
        draw.rectangle(xy=[(x0, y0), (x1, y1)], outline="black", fill=colors[col_idx])

        # Wrap header text and calculate height
        lines = textwrap.wrap(header, width=10)  # Adjust the width as needed
        header_height = len(lines) * (TITLE_FONT_SIZE + 4)  # Adjust the line spacing

        # Calculate y-coordinate for multiline header
        y_header = y0 + (CELL_HEIGHT - header_height) / 2  # Adjust the y-coordinate for centering
        for line_idx, line in enumerate(lines):
            y_text = y_header + line_idx * (TITLE_FONT_SIZE + 4)  # Adjust the y-coordinate for each line
            draw.text((x0 + 5, y_text), line, font=title_font, fill="black", align="center")

    # Draw data rows
    for row_idx in range(num_rows - 1):
        y0 = (row_idx + 1) * CELL_HEIGHT + TABLE_PADDING

        for col_idx, col_name in enumerate(column_headers):
            if col_name == "Stock":
                value = str(descriptions[row_idx])
            else:
                value = round(table_data.values[row_idx][col_idx - 1], 2)
            x0 = (col_idx + 0) * CELL_WIDTH + TABLE_PADDING
            y1 = (row_idx + 2) * CELL_HEIGHT + TABLE_PADDING
            x1 = (col_idx + 1) * CELL_WIDTH + TABLE_PADDING

            # Wrap text and calculate height
            if str(value) == 'nan':
                value = ""
            lines = textwrap.wrap(str(value), width=15)  # Adjust the width as needed
            cell_height = len(lines) * (VALUE_FONT_SIZE + 4)  # Adjust the line spacing

            # cell_height = 0
            if str(value) != 'nan':
                draw.rectangle([(x0, y0), (x1, y1 + cell_height)], outline="black", fill=colors[col_idx])
            else:
                draw.rectangle([(x0, y0), (x1, y1 + cell_height)], outline="black", fill="white")

            # Calculate y-coordinate for multiline text
            y_text = y0 + (cell_height - len(lines) * (VALUE_FONT_SIZE + 4)) / 2  # Adjusts y-coordinate for centering

            # Draw multiline text
            multiline_text = "\n".join(lines)
            draw.multiline_text(
                (x0 + 5, y_text), multiline_text, font=value_font, fill="black", align="center", spacing=4
            )
    # Save the image Table
    image.save(f"{path} {ending_file_name}.png")
