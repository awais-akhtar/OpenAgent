"""Curated metadata and integrity-checked imports for public materials data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .dataset import analyse_dataset_upload, parse_dataset_upload


FetchBytes = Callable[[str, int], bytes]


@dataclass(frozen=True)
class PublicMaterialsDataset:
    id: str
    title: str
    description: str
    rows: int
    target: str
    download_url: str
    source_url: str
    sha256: str
    license_id: str
    license_url: str
    citation: str
    file_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Metadata is derived from the upstream matminer registry. Dataset bytes are
# never bundled; each download is checked against the upstream SHA-256 value.
PUBLIC_MATERIALS_CATALOG: tuple[PublicMaterialsDataset, ...] = (
    PublicMaterialsDataset(
        id="dielectric-constant-2017",
        title="Dielectric Constant",
        description="DFPT-PBE dielectric properties for 1,056 inorganic structures.",
        rows=1056,
        target="n",
        download_url="https://ndownloader.figshare.com/files/13213475",
        source_url="https://doi.org/10.1038/sdata.2016.134",
        sha256="8eb24812148732786cd7c657eccfc6b5ee66533429c2cfbcc4f0059c0295e8b6",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Petousis et al., Scientific Data 4, 160134 (2017).",
        file_name="dielectric_constant.json.gz",
    ),
    PublicMaterialsDataset(
        id="elastic-tensor-2015",
        title="Elastic Tensor 2015",
        description="DFT-PBE elastic properties for 1,181 inorganic structures.",
        rows=1181,
        target="K_VRH",
        download_url="https://ndownloader.figshare.com/files/13220603",
        source_url="https://doi.org/10.1038/sdata.2015.9",
        sha256="8c3b342f75da7e7baa1b769b59554485fd647f4cb1da9318d0d1ba3b3b838183",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="de Jong et al., Scientific Data 2, 150009 (2015).",
        file_name="elastic_tensor_2015.json.gz",
    ),
    PublicMaterialsDataset(
        id="formation-enthalpy-2017",
        title="Experimental Formation Enthalpy",
        description="Experimental formation enthalpies and matched computed values for 1,276 inorganic compounds.",
        rows=1276,
        target="e_form expt",
        download_url="https://ndownloader.figshare.com/files/13464467",
        source_url="https://doi.org/10.1038/sdata.2017.162",
        sha256="cb0e1bba0de70ca977a3a6b3c0020cc986fe32affc90fc36752beb61aaf6f5b4",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Kim et al., Scientific Data 4, 170162 (2017).",
        file_name="expt_formation_enthalpy.json.gz",
    ),
)


def list_public_materials_datasets() -> list[dict[str, Any]]:
    return [{**entry.to_dict(), "network_required": True, "bundled_records": False} for entry in PUBLIC_MATERIALS_CATALOG]


def import_public_materials_dataset(
    dataset_id: str,
    *,
    allow_external: bool = False,
    license_accepted: bool = False,
    fetcher: FetchBytes | None = None,
    catalog: Iterable[PublicMaterialsDataset] = PUBLIC_MATERIALS_CATALOG,
    max_download_bytes: int = 128 * 1024 * 1024,
) -> dict[str, Any]:
    """Fetch one catalog entry after explicit approval and verify its identity."""

    entry = _find_entry(dataset_id, catalog)
    if not allow_external:
        raise PermissionError("public dataset import requires allow_external=true")
    if not license_accepted:
        raise PermissionError(f"accept the {entry.license_id} dataset license before import")
    parsed = urlparse(entry.download_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("catalog downloads must use HTTPS")
    content = (fetcher or _fetch_https)(entry.download_url, max_download_bytes)
    if len(content) > max_download_bytes:
        raise ValueError("materials dataset download exceeds the configured size limit")
    digest = hashlib.sha256(content).hexdigest()
    if digest != entry.sha256:
        raise ValueError(f"materials dataset checksum mismatch: expected {entry.sha256}, received {digest}")
    records = parse_dataset_upload(content, name=entry.file_name, target=entry.target)
    if len(records) != entry.rows:
        raise ValueError(f"materials dataset row count mismatch: expected {entry.rows}, received {len(records)}")
    return {
        "status": "imported",
        "records": records,
        "target": entry.target,
        "provenance": {
            **entry.to_dict(),
            "verified_sha256": digest,
            "license_accepted": True,
            "network_approved": True,
            "retrieved_bytes": len(content),
        },
        "privacy_boundary": "The approved HTTPS download is the only network operation. Parsing and analysis run locally.",
    }


def analyse_public_materials_dataset(
    dataset_id: str,
    *,
    allow_external: bool = False,
    license_accepted: bool = False,
    fetcher: FetchBytes | None = None,
    catalog: Iterable[PublicMaterialsDataset] = PUBLIC_MATERIALS_CATALOG,
    artifact_dir: str | None = None,
    seed: int = 31,
) -> dict[str, Any]:
    imported = import_public_materials_dataset(
        dataset_id,
        allow_external=allow_external,
        license_accepted=license_accepted,
        fetcher=fetcher,
        catalog=catalog,
    )
    provenance = imported["provenance"]
    raw_records = imported["records"]
    # Re-encode validated rows to keep the established analysis path identical
    # for local uploads and catalog imports.
    encoded = __import__("json").dumps(raw_records, separators=(",", ":"), default=str).encode("utf-8")
    report = analyse_dataset_upload(
        encoded,
        name=f"{dataset_id}.json",
        target=str(imported["target"]),
        artifact_dir=artifact_dir,
        test_fraction=0.2,
        seed=seed,
    )
    report["provenance"] = provenance
    report["split"] = {"method": "seeded-random", "training_fraction": 0.8, "test_fraction": 0.2, "seed": seed}
    report["privacy_boundary"] = imported["privacy_boundary"]
    return report


def _find_entry(dataset_id: str, catalog: Iterable[PublicMaterialsDataset]) -> PublicMaterialsDataset:
    normalized = dataset_id.strip().lower()
    for entry in catalog:
        if entry.id == normalized:
            return entry
    raise KeyError(f"unknown public materials dataset: {dataset_id}")


def _fetch_https(url: str, max_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "OpenAgent materials catalog/1"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - catalog URL and approval are both enforced
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError("materials dataset download exceeds the configured size limit")
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("materials dataset download exceeds the configured size limit")
    return content
