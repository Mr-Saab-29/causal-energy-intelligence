from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pandas as pd
import pytest

from src.data.causal_archive_compact import (
    SupabaseStorage,
    archive_and_compact_month,
    build_manifest,
    normalize_parquet_values,
    sha256_file,
)


def test_normalize_parquet_values_converts_uuid_objects() -> None:
    identifier = uuid4()
    frame = pd.DataFrame({"id": [identifier], "value": [1.5]})

    result = normalize_parquet_values(frame)

    assert result.loc[0, "id"] == str(identifier)
    assert result.loc[0, "value"] == 1.5


def test_manifest_uses_portable_filenames(tmp_path: Path) -> None:
    archive = tmp_path / "cross_border_observations.parquet"
    archive.write_bytes(b"archive")
    files = [
        {
            "table": "cross_border_observations",
            "path": str(archive),
            "row_count": 2,
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    ]

    manifest = build_manifest(
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-09-01", tz="UTC"),
        "historical_final",
        files,
        {"causal_cross_border_hourly_features": 2},
    )

    assert manifest["files"][0]["file"] == archive.name
    assert "path" not in manifest["files"][0]


def test_purge_requires_verified_storage_before_database_access() -> None:
    with pytest.raises(ValueError, match="verified Supabase Storage upload"):
        archive_and_compact_month(
            object(),  # type: ignore[arg-type]
            pd.Timestamp("2026-08-01", tz="UTC"),
            pd.Timestamp("2026-09-01", tz="UTC"),
            purge_raw=True,
        )


def test_storage_upload_is_downloaded_and_checksum_verified(tmp_path: Path) -> None:
    payload = b"verified archive"
    archive = tmp_path / "part.parquet"
    archive.write_bytes(payload)
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path.endswith("/bucket/causal-grid-archive"):
            return httpx.Response(
                400,
                json={
                    "statusCode": "404",
                    "code": "NoSuchBucket",
                    "message": "Bucket not found",
                },
            )
        if request.url.path.endswith("/bucket"):
            body = json.loads(request.content)
            assert body["public"] is False
            return httpx.Response(200, json={"name": body["name"]})
        if "/object/authenticated/" in request.url.path:
            return httpx.Response(200, content=payload)
        if "/object/" in request.url.path:
            assert request.headers["x-upsert"] == "true"
            assert request.content == payload
            return httpx.Response(200, json={"Key": "archive"})
        return httpx.Response(500)

    storage = SupabaseStorage("https://project.example", "secret")
    storage.client.close()
    storage.client = httpx.Client(
        base_url="https://project.example",
        transport=httpx.MockTransport(handler),
    )

    storage.ensure_private_bucket()
    storage.upload_verified(archive, "historical_final/2026-08/part.parquet", sha256_file(archive))
    storage.close()

    assert requests == [
        ("GET", "/storage/v1/bucket/causal-grid-archive"),
        ("POST", "/storage/v1/bucket"),
        (
            "POST",
            "/storage/v1/object/causal-grid-archive/historical_final/2026-08/part.parquet",
        ),
        (
            "GET",
            "/storage/v1/object/authenticated/causal-grid-archive/historical_final/2026-08/part.parquet",
        ),
    ]


def test_storage_error_does_not_include_credentials() -> None:
    secret = "do-not-expose"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=f"request rejected for {secret}")

    storage = SupabaseStorage("https://project.example", secret)
    storage.client.close()
    storage.client = httpx.Client(
        base_url="https://project.example",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError) as error:
        storage.ensure_private_bucket()
    storage.close()

    assert secret not in str(error.value)


def test_new_secret_key_is_not_sent_as_bearer_token() -> None:
    storage = SupabaseStorage("https://project.example", "sb_secret_example")

    assert storage.client.headers["apikey"] == "sb_secret_example"
    assert "authorization" not in storage.client.headers
    storage.close()


def test_legacy_service_role_jwt_is_sent_as_bearer_token() -> None:
    storage = SupabaseStorage("https://project.example", "eyJlegacy")

    assert storage.client.headers["apikey"] == "eyJlegacy"
    assert storage.client.headers["authorization"] == "Bearer eyJlegacy"
    storage.close()


def test_missing_storage_object_is_reported_as_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"statusCode": "404", "code": "NoSuchKey"},
        )

    storage = SupabaseStorage("https://project.example", "sb_secret_example")
    storage.client.close()
    storage.client = httpx.Client(
        base_url="https://project.example",
        transport=httpx.MockTransport(handler),
    )

    assert storage.object_exists("historical_final/2026-06/manifest.json") is False
    storage.close()
