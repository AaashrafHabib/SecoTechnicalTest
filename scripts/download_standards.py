"""Download publicly available EU construction standard documents.

Downloads freely available Eurocode guidance documents from the EU Joint Research
Centre (JRC) publications portal. All documents are published under CC BY 4.0
and hosted on official EU domains (eurocodes.jrc.ec.europa.eu).
"""

import ssl
import urllib.request
from pathlib import Path

DATA_DIR = Path("data/raw")

DOCUMENTS = [
    {
        "url": "https://eurocodes.jrc.ec.europa.eu/sites/default/files/2023-01/JRC131689_01.pdf",
        "filename": "jrc_fire_safety_engineering_eurocodes.pdf",
        "title": "Fire Safety Engineering in Europe - Status and Implementation (JRC, 2023)",
    },
    {
        "url": "https://eurocodes.jrc.ec.europa.eu/sites/default/files/2022-11/JRC%20Report_%20Tunnels%20design%20and%20Eurocodes_2022_11_02%20(no%20Annex%20C).pdf",
        "filename": "jrc_tunnels_design_eurocodes.pdf",
        "title": "Tunnels and Underground Structures Design - Eurocodes (JRC, 2022)",
    },
    {
        "url": "https://eurocodes.jrc.ec.europa.eu/sites/default/files/2021-12/JRC%20report%20existing%20structures%20ff%20online.pdf",
        "filename": "jrc_assessment_retrofitting_structures.pdf",
        "title": "Assessment and Retrofitting of Existing Structures (JRC, 2015)",
    },
    {
        "url": "https://eurocodes.jrc.ec.europa.eu/sites/default/files/2021-12/JRC97893_State_of_Eurocodes_implementation_version_online.pdf",
        "filename": "jrc_state_eurocodes_implementation.pdf",
        "title": "State of Implementation of the Eurocodes in the EU (JRC, 2015)",
    },
    {
        "url": "https://eurocodes.jrc.ec.europa.eu/sites/default/files/2021-12/JRC113687_jrc_new_reliability_reportprint-f_1.pdf",
        "filename": "jrc_reliability_structural_members_eurocodes.pdf",
        "title": "Reliability of Structural Members Designed with Eurocodes NDPs (JRC, 2019)",
    },
]


def download_documents() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    downloaded = 0
    for doc in DOCUMENTS:
        filepath = DATA_DIR / doc["filename"]
        if filepath.exists():
            print(f"Already exists: {doc['filename']}")
            downloaded += 1
            continue

        print(f"Downloading: {doc['title']}")
        print(f"  URL: {doc['url']}")

        try:
            req = urllib.request.Request(
                doc["url"],
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/pdf,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=180, context=ctx) as response:
                content = response.read()

            if len(content) < 50000:
                print(f"  WARNING: File too small ({len(content)} bytes), skipping")
                continue

            filepath.write_bytes(content)
            size_mb = len(content) / (1024 * 1024)
            print(f"  Saved: {filepath} ({size_mb:.1f} MB)")
            downloaded += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {downloaded}/{len(DOCUMENTS)} documents in {DATA_DIR}/")


if __name__ == "__main__":
    download_documents()
