from PIL import Image, ImageDraw, ImageFont

# Global constants
CELL_WIDTH: list[int, int, int] = [200, 105, 1255]
CELL_HEIGHT = 40
TABLE_PADDING = 10


# def draw_all_and_save_as_png(file_name: str, symbols: list[str], values: list[float], descriptions: list[str],
#                              header_text, percent_mode: bool = True) -> None:
#     """
#     Stocks Representation
#     """
#     if len(values) != len(symbols) or len(values) != len(descriptions):
#         raise ValueError("Input lists must have the same length.")
#     image, draw = create_blank_image(num_rows=len(symbols))
#     draw_table_header(draw=draw, header_text=header_text)
#     draw_table_rows(symbols=symbols, values=values, descriptions=descriptions,
#                     draw=draw, percent_mode=percent_mode)
#     image.save(f"{file_name}.png")  # Save the image


def create_blank_image(num_rows: int) -> tuple[Image, ImageDraw]:
    # Calculate image size
    image_width = CELL_WIDTH[0] + CELL_WIDTH[1] + CELL_WIDTH[2] + 2 * TABLE_PADDING
    image_height = (num_rows + 1) * CELL_HEIGHT + 2 * TABLE_PADDING

    # Create a blank image
    image = Image.new(mode="RGB", size=(image_width, image_height), color="white")
    draw = ImageDraw.Draw(image)
    return image, draw


def draw_table_header(draw: ImageDraw, header_text: ImageFont.truetype) -> None:
    header_font = ImageFont.truetype("arialbd.ttf", size=26)
    # header_text: list[str, str, str] = ['Stock', 'Weight', 'Description']
    for col_idx in range(len(CELL_WIDTH)):
        x0: int = calculate_cumulative_col_offset(col_idx) + TABLE_PADDING
        y0: int = TABLE_PADDING
        x1: int = x0 + CELL_WIDTH[col_idx]
        y1: int = y0 + CELL_HEIGHT
        draw.rectangle(xy=[(x0, y0), (x1, y1)], fill="lightgray", outline="black")
        draw.text(xy=(x0 + 5, y0 + 5), text=header_text[col_idx], fill="black", font=header_font)


def draw_table_rows(symbols: list[str], values: list[float], descriptions: list[str], draw: ImageDraw,
                    percent_mode: bool) -> None:
    x0: list[int, int, int] = [
        calculate_cumulative_col_offset(0) + TABLE_PADDING,
        calculate_cumulative_col_offset(1) + TABLE_PADDING,
        calculate_cumulative_col_offset(2) + TABLE_PADDING,
    ]
    x1: list[int, int, int] = [x0[0] + CELL_WIDTH[0], x0[1] + CELL_WIDTH[1], x0[2] + CELL_WIDTH[2]]
    row_font = ImageFont.truetype("ariali.ttf", size=26)
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


def calculate_cumulative_col_offset(col_idx: int) -> int:
    res_sum: int = 0
    for i in range(col_idx):
        res_sum += CELL_WIDTH[i]
    return res_sum
