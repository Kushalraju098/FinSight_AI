# sec_html_loader.py

import re
from pathlib import Path

from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class SECHTMLLoader(BaseLoader):

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    # =========================================================
    # READ HTML FILE
    # =========================================================

    def _read_html(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    # =========================================================
    # CLEAN INDIVIDUAL TABLE CELL
    # =========================================================

    def clean_cell(self, value):

        if value is None:
            return ""

        value = str(value)

        # Replace non-breaking spaces
        value = value.replace("\xa0", " ")

        # Remove excessive whitespace
        value = re.sub(r"\s+", " ", value)

        value = value.strip()

        # -----------------------------------------------------
        # Convert numbers such as:
        # 2025.0 -> 2025
        # 178353.0 -> 178353
        # -----------------------------------------------------

        if re.fullmatch(r"-?\d+\.0", value):
            value = value[:-2]

        # -----------------------------------------------------
        # Convert currency numbers such as:
        # $178353.0 -> $178353
        # -----------------------------------------------------

        match = re.fullmatch(r"(\$)\s*(-?\d+)\.0", value)

        if match:
            value = f"{match.group(1)}{match.group(2)}"

        return value

    # =========================================================
    # CHECK WHETHER TWO VALUES REPRESENT SAME NUMBER
    # =========================================================

    def same_numeric_value(self, value1, value2):

        if not value1 or not value2:
            return False

        v1 = value1.replace(",", "").replace("$", "").strip()
        v2 = value2.replace(",", "").replace("$", "").strip()

        try:
            return float(v1) == float(v2)

        except ValueError:
            return False

    # =========================================================
    # CONVERT HTML TABLE INTO POSITIONAL GRID
    # =========================================================

    def html_table_to_grid(self, table):

        rows = table.find_all("tr")

        grid = []

        # Track cells occupied by rowspan
        occupied = {}

        for row_index, row in enumerate(rows):

            cells = row.find_all(["td", "th"])

            current_row = []

            col_index = 0

            for cell in cells:

                # -------------------------------------------------
                # Find next free column
                # -------------------------------------------------

                while (row_index, col_index) in occupied:
                    current_row.append(occupied[(row_index, col_index)])
                    col_index += 1

                value = self.clean_cell(cell.get_text(" ", strip=True))

                # -------------------------------------------------
                # Get colspan / rowspan
                # -------------------------------------------------

                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))

                # -------------------------------------------------
                # Add cell according to colspan
                # -------------------------------------------------

                for span_col in range(colspan):

                    current_row.append(value)

                    # ------------------------------------------------
                    # Handle rowspan
                    # ------------------------------------------------

                    if rowspan > 1:

                        for span_row in range(1, rowspan):

                            occupied[
                                (row_index + span_row, col_index + span_col)
                            ] = value

                col_index += colspan

            # -----------------------------------------------------
            # Fill remaining occupied cells
            # -----------------------------------------------------

            while (row_index, col_index) in occupied:

                current_row.append(
                    occupied[(row_index, col_index)]
                )

                col_index += 1

            grid.append(current_row)

        # ---------------------------------------------------------
        # Make all rows same length
        # ---------------------------------------------------------

        if grid:

            max_columns = max(len(row) for row in grid)

            for row in grid:

                while len(row) < max_columns:
                    row.append("")

        return grid

    # =========================================================
    # REMOVE DUPLICATE NUMERIC CELLS
    # =========================================================

    def remove_duplicate_numeric_cells(self, row):

        cleaned = row.copy()

        for i in range(len(cleaned) - 1):

            current = cleaned[i]
            next_value = cleaned[i + 1]

            if self.same_numeric_value(current, next_value):

                # IMPORTANT:
                # Do NOT delete the cell.
                # Blank it so column positions remain unchanged.

                cleaned[i + 1] = ""

        return cleaned

    # =========================================================
    # REMOVE DUPLICATE TEXT CELLS
    # =========================================================

    def remove_duplicate_text_cells(self, row):

        cleaned = row.copy()

        for i in range(len(cleaned) - 1):

            current = cleaned[i].strip()
            next_value = cleaned[i + 1].strip()

            if (
                current
                and next_value
                and current.lower() == next_value.lower()
            ):

                # Keep the original position
                cleaned[i + 1] = ""

        return cleaned

    # =========================================================
    # COMBINE CURRENCY CELLS
    # =========================================================

    def combine_currency_cells(self, row):

        cleaned = row.copy()

        for i in range(len(cleaned) - 1):

            current = cleaned[i].strip()
            next_value = cleaned[i + 1].strip()

            # Example:
            #
            # $
            # 178353
            #
            # becomes:
            #
            # $178353
            #

            if current == "$" and next_value:

                cleaned[i] = f"${next_value}"
                cleaned[i + 1] = ""

        return cleaned

    # =========================================================
    # COMBINE PERCENTAGE CELLS
    # =========================================================

    def combine_percentage_cells(self, row):

        cleaned = row.copy()

        for i in range(len(cleaned) - 1):

            current = cleaned[i].strip()
            next_value = cleaned[i + 1].strip()

            # Example:
            #
            # 7
            # %
            #
            # becomes:
            #
            # 7%
            #

            if (
                current
                and next_value == "%"
            ):

                cleaned[i] = f"{current}%"
                cleaned[i + 1] = ""

        return cleaned

    # =========================================================
    # NORMALIZE ONE TABLE ROW
    # =========================================================

    def normalize_row(self, row):

        row = [
            self.clean_cell(value)
            for value in row
        ]

        # Remove duplicated representations
        row = self.remove_duplicate_numeric_cells(row)

        # Remove duplicated text representations
        row = self.remove_duplicate_text_cells(row)

        # Combine currency
        row = self.combine_currency_cells(row)

        # Combine percentage
        row = self.combine_percentage_cells(row)

        return row

    # =========================================================
    # NORMALIZE COMPLETE TABLE
    # =========================================================

    def normalize_table(self, grid):

        normalized = []

        for row in grid:

            normalized_row = self.normalize_row(row)

            normalized.append(normalized_row)

        return normalized

    # =========================================================
    # REMOVE COMPLETELY EMPTY ROWS
    # =========================================================

    def remove_empty_rows(self, grid):

        cleaned = []

        for row in grid:

            if any(cell.strip() for cell in row):

                cleaned.append(row)

        return cleaned

    # =========================================================
    # REMOVE COMPLETELY EMPTY COLUMNS
    # =========================================================

    def remove_empty_columns(self, grid):

        if not grid:
            return grid

        max_columns = max(len(row) for row in grid)

        # Make sure every row has same number of columns
        for row in grid:

            while len(row) < max_columns:
                row.append("")

        columns_to_keep = []

        for col_index in range(max_columns):

            has_value = any(
                grid[row_index][col_index].strip()
                for row_index in range(len(grid))
            )

            if has_value:
                columns_to_keep.append(col_index)

        cleaned = []

        for row in grid:

            cleaned.append(
                [
                    row[col_index]
                    for col_index in columns_to_keep
                ]
            )

        return cleaned

    # =========================================================
    # CONVERT TABLE TO MARKDOWN
    # =========================================================

    def table_to_markdown(self, grid):

        if not grid:
            return ""

        grid = self.remove_empty_rows(grid)

        grid = self.remove_empty_columns(grid)

        if not grid:
            return ""

        # -----------------------------------------------------
        # Create Markdown manually.
        #
        # This avoids pandas automatically interpreting
        # the first row as a header.
        # -----------------------------------------------------

        markdown_lines = []

        for row_index, row in enumerate(grid):

            row_text = "| " + " | ".join(row) + " |"

            markdown_lines.append(row_text)

            # Add separator after first row
            if row_index == 0:

                separator = (
                    "| "
                    + " | ".join("---" for _ in row)
                    + " |"
                )

                markdown_lines.append(separator)

        return "\n".join(markdown_lines)

    # =========================================================
    # REMOVE SEC PAGE ARTIFACTS
    # =========================================================

    def remove_sec_page_artifacts(self, text):

        lines = text.splitlines()

        cleaned = []

        for line in lines:

            line = line.strip()

            # -------------------------------------------------
            # Remove empty lines
            # -------------------------------------------------

            if not line:
                continue

            # -------------------------------------------------
            # Remove "Table of Contents"
            # -------------------------------------------------

            if line.lower() == "table of contents":
                continue

            # -------------------------------------------------
            # Remove standalone page numbers
            #
            # Example:
            # 26
            # 27
            # 28
            # -------------------------------------------------

            if re.fullmatch(r"\d{1,3}", line):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    # =========================================================
    # EXTRACT NARRATIVE TEXT
    # =========================================================

    def extract_narrative_text(self, html):

        soup = BeautifulSoup(html, "lxml")

        # =====================================================
        # REMOVE TABLES
        #
        # Tables are extracted separately.
        # =====================================================

        for table in soup.find_all("table"):

            table.decompose()

        # =====================================================
        # REMOVE SCRIPT / STYLE / NOSCRIPT
        # =====================================================

        for tag in soup.find_all(
            ["script", "style", "noscript"]
        ):

            tag.decompose()

        # =====================================================
        # REMOVE INLINE XBRL ELEMENTS
        # =====================================================

        for tag in soup.find_all():

            tag_name = tag.name.lower()

            if (
                tag_name.startswith("ix:")
                or tag_name.startswith("xbrli:")
                or tag_name.startswith("xbrldi:")
                or tag_name.startswith("dei:")
            ):

                tag.decompose()

        # =====================================================
        # EXTRACT TEXT
        # =====================================================

        text = soup.get_text("\n")

        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")

        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize newlines
        text = re.sub(r" *\n *", "\n", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        # =====================================================
        # REMOVE XBRL / NAMESPACE GARBAGE
        # =====================================================

        cleaned_lines = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # -------------------------------------------------
            # Remove namespace / XBRL lines
            #
            # us-gaap:
            # xbrli:
            # xbrldi:
            # dei:
            # iso4217:
            # utr:
            # -------------------------------------------------

            if re.match(
                r"^(us-gaap:|xbrli:|xbrldi:|dei:|iso4217:|utr:)",
                line,
                re.IGNORECASE
            ):

                continue

            # -------------------------------------------------
            # Generic namespace pattern
            # -------------------------------------------------

            if re.match(
                r"^[a-zA-Z][a-zA-Z0-9_-]*:",
                line
            ):

                continue

            # -------------------------------------------------
            # Remove 10-digit CIK values
            # -------------------------------------------------

            if re.fullmatch(r"\d{10}", line):

                continue

            # -------------------------------------------------
            # Remove duration values
            #
            # Example:
            # P1Y
            # P3M
            # -------------------------------------------------

            if re.fullmatch(
                r"P\d+[YMD]",
                line,
                re.IGNORECASE
            ):

                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # =====================================================
        # FIND ACTUAL SEC FILING START
        # =====================================================

        match = re.search(
            r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION",
            text,
            flags=re.IGNORECASE
        )

        if match:

            text = text[match.start():]

        # =====================================================
        # REMOVE SEC PAGE ARTIFACTS
        # =====================================================

        text = self.remove_sec_page_artifacts(text)

        # =====================================================
        # FINAL WHITESPACE CLEANUP
        # =====================================================

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # =========================================================
    # EXTRACT TABLES
    # =========================================================

    def extract_tables(self, html):

        soup = BeautifulSoup(html, "lxml")

        tables = soup.find_all("table")

        extracted_tables = []

        for table_number, table in enumerate(
            tables,
            start=1
        ):

            # -------------------------------------------------
            # Convert HTML table to positional grid
            # -------------------------------------------------

            grid = self.html_table_to_grid(table)

            # -------------------------------------------------
            # Normalize table
            # -------------------------------------------------

            grid = self.normalize_table(grid)

            # -------------------------------------------------
            # Remove empty rows
            # -------------------------------------------------

            grid = self.remove_empty_rows(grid)

            # -------------------------------------------------
            # Remove empty columns
            # -------------------------------------------------

            grid = self.remove_empty_columns(grid)

            # -------------------------------------------------
            # Skip empty tables
            # -------------------------------------------------

            if not grid:
                continue

            # -------------------------------------------------
            # Convert to Markdown
            # -------------------------------------------------

            markdown = self.table_to_markdown(grid)

            if not markdown.strip():
                continue

            extracted_tables.append(
                {
                    "table_number": table_number,
                    "content": markdown
                }
            )

        return extracted_tables

    # =========================================================
    # METADATA
    # =========================================================

    def get_metadata(self):

        filename = self.file_path.name

        company = self.file_path.parent.name

        # -----------------------------------------------------
        # Detect filing type
        # -----------------------------------------------------

        filename_upper = filename.upper()

        if "10-K" in filename_upper or "10K" in filename_upper:

            filing_type = "10-K"

        elif "10-Q" in filename_upper or "10Q" in filename_upper:

            filing_type = "10-Q"

        else:

            filing_type = "unknown"

        # -----------------------------------------------------
        # Detect filing year
        #
        # Example:
        # AAPL_2025_10K.html
        # -----------------------------------------------------

        year_match = re.search(
            r"(20\d{2})",
            filename
        )

        filing_year = (
            year_match.group(1)
            if year_match
            else None
        )

        return {
            "source": str(self.file_path),
            "company": company,
            "filing_type": filing_type,
            "filing_year": filing_year
        }

    # =========================================================
    # LAZY LOAD
    # =========================================================

    def lazy_load(self):

        # =====================================================
        # READ HTML
        # =====================================================

        html = self._read_html()

        # =====================================================
        # METADATA
        # =====================================================

        metadata = self.get_metadata()

        # =====================================================
        # EXTRACT NARRATIVE TEXT
        # =====================================================

        narrative_text = self.extract_narrative_text(html)

        # =====================================================
        # CREATE TEXT DOCUMENT
        # =====================================================

        if narrative_text:

            text_metadata = {
                **metadata,
                "content_type": "text"
            }

            yield Document(
                page_content=narrative_text,
                metadata=text_metadata
            )

        # =====================================================
        # EXTRACT TABLES
        # =====================================================

        tables = self.extract_tables(html)

        # =====================================================
        # CREATE TABLE DOCUMENTS
        # =====================================================

        for table in tables:

            table_metadata = {
                **metadata,
                "content_type": "table",
                "table_number": table["table_number"]
            }

            yield Document(
                page_content=table["content"],
                metadata=table_metadata
            )