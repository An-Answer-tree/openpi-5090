"""Export the editable slide figure to vector PDF and a preview PNG."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from pypdf import PdfReader
from pypdf import PdfWriter


def main() -> None:
    """Render all pages and scale the PDF to a 17 cm publication width."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pptx", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acpd_figure_pdf_") as temporary_dir:
        temporary = Path(temporary_dir)
        profile = temporary / "profile"
        pdf_filter = 'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}'
        subprocess.run(
            [
                "soffice",
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless",
                "--norestore",
                "--convert-to",
                pdf_filter,
                "--outdir",
                str(temporary),
                str(args.pptx.resolve()),
            ],
            check=True,
            timeout=60,
        )
        reader = PdfReader(temporary / f"{args.pptx.stem}.pdf")
        all_pages = PdfWriter()
        for page in reader.pages:
            page.scale_by((17 / 2.54 * 72) / float(page.mediabox.width))
            all_pages.add_page(page)
        all_pages.add_metadata({"/Title": "ACPD qualitative rollouts: main figure and view enlargements"})
        with (output / "ACPD_rollouts_all.pdf").open("wb") as stream:
            all_pages.write(stream)
        main_page = PdfWriter()
        main_page.add_page(reader.pages[0])
        main_page.add_metadata({"/Title": "ACPD rollouts across four views and four LIBERO suites"})
        with (output / "Figure4_ACPD_rollouts.pdf").open("wb") as stream:
            main_page.write(stream)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-scale-to",
            "2400",
            str(output / "Figure4_ACPD_rollouts.pdf"),
            str(output / "Figure4_ACPD_rollouts"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
