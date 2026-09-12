"""
Utility functions for building and merging PDF report pages.

Author: Caolan Rafferty
Date: 2023-07-02
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pdfrw
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from src.utils.types import Frame
from src.utils.util import convert_to_snake_case


def _df_to_pdf_page(
    title: str,
    df: Frame,
    page: int,
    output_dir: str,
    highlight_columns: Optional[List[str]],
    thresholds: Optional[List[float]],
    operators: Optional[List[str]],
    highlight_colour: Optional[str],
) -> str:
    """Render a single page of a (possibly split) DataFrame table and save it as a PDF."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", ha="left")
        else:
            cell.set_text_props(ha="left")
            if highlight_columns and thresholds and operators and highlight_colour:
                for i, col_name in enumerate(highlight_columns):
                    try:
                        col_index = df.columns.get_loc(col_name)
                    except KeyError:
                        raise ValueError(f"Column '{col_name}' not found in dataframe")
                    if col == col_index:
                        try:
                            cell_value = float(
                                cell.get_text().get_text().replace(",", "").rstrip("%")
                            )
                        except ValueError:
                            continue
                        if operators[i] == ">" and cell_value > thresholds[i]:
                            cell.set_facecolor(highlight_colour)
                        elif operators[i] == "<" and cell_value < thresholds[i]:
                            cell.set_facecolor(highlight_colour)

    ax.set_title(title, fontsize=12, fontweight="bold", y=0.9)
    file = f"{output_dir}/{convert_to_snake_case(title)}_{page}.pdf"
    pp = PdfPages(file)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()
    plt.close(fig)
    return file


def df_to_pdf(
    title: str,
    df: Frame,
    output_dir: str,
    highlight_columns: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    operators: Optional[List[str]] = None,
    highlight_colour: Optional[str] = None,
    max_rows: int = 14,
) -> List[str]:
    """
    Save a DataFrame as one or more PDF pages, with optional highlighting of cells.

    Rows beyond `max_rows` are split across additional, separately numbered pages.

    Parameters:
    title (str): Title of the PDF document.
    df (Frame): The DataFrame to be saved as a PDF.
    output_dir (str): The path to the output directory.
    highlight_columns (List[str]): List of column names to be highlighted. Defaults to None.
    thresholds (List[float]): List of threshold values for highlighting. Defaults to None.
    operators (List[str]): List of comparison operators ('>' or '<') for highlighting. Defaults to None.
    highlight_colour (str): The colour for highlighting the cells. Defaults to None.
    max_rows (int): Maximum number of rows per page. Defaults to 14.

    Returns:
    List[str]: The file path(s) of the created PDF(s), one per page.
    """
    if len(df) <= max_rows:
        return [
            _df_to_pdf_page(
                title,
                df,
                1,
                output_dir,
                highlight_columns,
                thresholds,
                operators,
                highlight_colour,
            )
        ]

    dfs = np.array_split(df, np.ceil(len(df) / max_rows))
    return [
        _df_to_pdf_page(
            title,
            sub_df,
            i + 1,
            output_dir,
            highlight_columns,
            thresholds,
            operators,
            highlight_colour,
        )
        for i, sub_df in enumerate(dfs)
    ]


def merge_pdfs(input_files: List[str], output_file: str) -> None:
    """
    Merge multiple PDF files into a single PDF file.

    Parameters:
    input_files (List[str]): A list of input file paths (strings) representing the PDF files to be merged.
    output_file (str): The output file path (string) where the merged PDF file will be saved.
    """
    pdf_output = pdfrw.PdfWriter()
    for file_name in input_files:
        pdf_input = pdfrw.PdfReader(file_name)
        for page in pdf_input.pages:
            pdf_output.addpage(page)
        os.remove(file_name)
    pdf_output.write(output_file)


def save_paragraphs_to_pdf(
    title: str, headings: List[str], paragraphs: List[str], output_dir: str
) -> str:
    """
    Save paragraphs to a PDF file with specified title, headings, and output file.

    Parameters:
    title (str): The main title of the document.
    headings (List[str]): A list of heading strings.
    paragraphs (List[str]): A list of paragraph strings.
    output_dir (str): The path to the output directory.

    Returns:
    str: The file path of the created PDF.
    """
    file = f"{output_dir}/{convert_to_snake_case(title)}.pdf"
    doc = SimpleDocTemplate(file, pagesize=letter)
    styles = getSampleStyleSheet()
    main_title_style = ParagraphStyle(
        name="MainTitle", parent=styles["Heading1"], alignment=1, spaceAfter=24
    )

    title_style = styles["Heading2"]
    title_style.alignment = 0
    paragraph_style = styles["Normal"]
    paragraph_style.fontSize = 10
    paragraph_style.spaceAfter = 12
    elements = []
    main_title_element = Paragraph(title, main_title_style)
    elements.append(main_title_element)

    for heading, paragraph in zip(headings, paragraphs):
        title_element = Paragraph(heading, title_style)
        elements.append(title_element)
        paragraph_element = Paragraph(paragraph, paragraph_style)
        elements.append(paragraph_element)

    doc.build(elements)
    return file
