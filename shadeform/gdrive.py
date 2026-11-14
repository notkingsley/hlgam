"""Google Drive helpers for Shadeform remote training transfers."""

from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path
from typing import Sequence


FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ("https://www.googleapis.com/auth/drive",)


def build_service(service_account_key_path: Path | str):
    """Build a Google Drive v3 service from a service-account JSON key."""
    from google.oauth2 import service_account  # type: ignore[import-untyped]
    from googleapiclient.discovery import build  # type: ignore[import-untyped]

    credentials = service_account.Credentials.from_service_account_file(
        str(service_account_key_path),
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def get_or_create_folder(service, name: str, parent_id: str) -> str:
    """Return the child folder named *name* under *parent_id*, creating it if needed."""
    escaped_name = name.replace("'", r"\'")
    query = (
        f"name='{escaped_name}' and mimeType='{FOLDER_MIME}' "
        f"and '{parent_id}' in parents and trashed=false"
    )
    response = (
        service.files()
        .list(
            q=query,
            fields="files(id)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        .execute()
    )
    files = response.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
    created = (
        service.files()
        .create(body=metadata, fields="id", supportsAllDrives=True)
        .execute()
    )
    return created["id"]


def find_folder(service, name: str, parent_id: str) -> str | None:
    """Return the child folder named *name* under *parent_id*, if it exists."""
    escaped_name = name.replace("'", r"\'")
    query = (
        f"name='{escaped_name}' and mimeType='{FOLDER_MIME}' "
        f"and '{parent_id}' in parents and trashed=false"
    )
    response = (
        service.files()
        .list(
            q=query,
            fields="files(id)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        .execute()
    )
    files = response.get("files", [])
    return files[0]["id"] if files else None


def list_children(service, folder_id: str) -> list[dict[str, str]]:
    """Return all non-trashed direct children of a Drive folder."""
    children: list[dict[str, str]] = []
    page_token: str | None = None
    while True:
        list_kwargs = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id,name,mimeType)",
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if page_token is not None:
            list_kwargs["pageToken"] = page_token
        response = (
            service.files()
            .list(**list_kwargs)
            .execute()
        )
        children.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if page_token is None:
            return children


def download_file(service, file_id: str, destination: Path) -> None:
    """Download one Google Drive file to *destination*."""
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import-untyped]

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with destination.open("wb") as file_handle:
        downloader = MediaIoBaseDownload(file_handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(f"Downloaded Drive file {file_id} -> {destination}", file=sys.stderr)


def upload_file(service, local_path: Path, parent_id: str) -> None:
    """Upload one local file into a Google Drive folder."""
    from googleapiclient.http import MediaFileUpload  # type: ignore[import-untyped]

    mime_type, _ = mimetypes.guess_type(str(local_path))
    metadata = {"name": local_path.name, "parents": [parent_id]}
    media = MediaFileUpload(
        str(local_path),
        mimetype=mime_type or "application/octet-stream",
        resumable=True,
    )
    service.files().create(
        body=metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    print(f"Uploaded {local_path}", file=sys.stderr)


def upload_tree(service, local_dir: Path, parent_id: str) -> None:
    """Upload the contents of *local_dir* recursively into a Drive folder."""
    for child in sorted(local_dir.iterdir()):
        if child.is_dir():
            folder_id = get_or_create_folder(service, child.name, parent_id)
            upload_tree(service, child, folder_id)
        elif child.is_file():
            upload_file(service, child, parent_id)


def upload_paths(
    service,
    *,
    root: Path,
    paths: Sequence[str],
    parent_folder_id: str,
    subfolder_name: str,
) -> list[str]:
    """Upload root-relative paths into a named Drive subfolder."""
    destination_folder_id = get_or_create_folder(service, subfolder_name, parent_folder_id)
    uploaded: list[str] = []
    for path_text in paths:
        relative_path = path_text.strip("/")
        local_path = root / relative_path
        if not local_path.exists():
            print(f"Skipping missing path: {local_path}", file=sys.stderr)
            continue
        if local_path.is_dir():
            child_folder_id = get_or_create_folder(service, local_path.name, destination_folder_id)
            upload_tree(service, local_path, child_folder_id)
        else:
            upload_file(service, local_path, destination_folder_id)
        uploaded.append(relative_path)
    return uploaded


def download_tree(service, folder_id: str, destination: Path) -> None:
    """Download a Drive folder's contents recursively into *destination*."""
    destination.mkdir(parents=True, exist_ok=True)
    for child in list_children(service, folder_id):
        child_destination = destination / child["name"]
        if child["mimeType"] == FOLDER_MIME:
            download_tree(service, child["id"], child_destination)
        else:
            download_file(service, child["id"], child_destination)


def download_paths(
    service,
    *,
    parent_folder_id: str,
    subfolder_name: str,
    destination: Path,
    paths: Sequence[str] | None = None,
) -> list[str]:
    """Download selected return paths from a named Drive subfolder."""
    source_folder_id = find_folder(service, subfolder_name, parent_folder_id)
    if source_folder_id is None:
        raise FileNotFoundError(
            f"Drive subfolder {subfolder_name!r} was not found under parent {parent_folder_id!r}"
        )
    if not paths:
        download_tree(service, source_folder_id, destination)
        return ["."]

    children = {child["name"]: child for child in list_children(service, source_folder_id)}
    downloaded: list[str] = []
    for path_text in paths:
        relative_path = path_text.strip("/")
        if not relative_path:
            continue
        drive_name = Path(relative_path).name
        child = children.get(drive_name)
        if child is None:
            print(f"Skipping missing Drive return path: {drive_name}", file=sys.stderr)
            continue
        local_destination = destination / relative_path
        if child["mimeType"] == FOLDER_MIME:
            download_tree(service, child["id"], local_destination)
        else:
            download_file(service, child["id"], local_destination)
        downloaded.append(relative_path)
    return downloaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Google Drive transfer helper for Shadeform jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-file", help="Download one Drive file by ID.")
    download_parser.add_argument("--sa-key", type=Path, required=True)
    download_parser.add_argument("--file-id", required=True)
    download_parser.add_argument("--dest", type=Path, required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload local paths into a Drive subfolder.")
    upload_parser.add_argument("--sa-key", type=Path, required=True)
    upload_parser.add_argument("--parent", required=True, help="Destination parent Drive folder ID.")
    upload_parser.add_argument("--subfolder", required=True, help="Run-specific Drive subfolder name.")
    upload_parser.add_argument("--root", type=Path, required=True, help="Local root for --path values.")
    upload_parser.add_argument("--path", dest="paths", action="append", required=True)

    download_paths_parser = subparsers.add_parser(
        "download",
        help="Download paths from a named Drive subfolder.",
    )
    download_paths_parser.add_argument("--sa-key", type=Path, required=True)
    download_paths_parser.add_argument("--parent", required=True, help="Source parent Drive folder ID.")
    download_paths_parser.add_argument("--subfolder", required=True, help="Run-specific Drive subfolder name.")
    download_paths_parser.add_argument("--dest", type=Path, required=True, help="Local destination root.")
    download_paths_parser.add_argument("--path", dest="paths", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = build_service(args.sa_key)
    if args.command == "download-file":
        download_file(service, args.file_id, args.dest)
        return 0
    if args.command == "upload":
        uploaded = upload_paths(
            service,
            root=args.root,
            paths=args.paths,
            parent_folder_id=args.parent,
            subfolder_name=args.subfolder,
        )
        print(f"Uploaded {len(uploaded)} path(s)", file=sys.stderr)
        return 0
    if args.command == "download":
        try:
            downloaded = download_paths(
                service,
                parent_folder_id=args.parent,
                subfolder_name=args.subfolder,
                destination=args.dest,
                paths=args.paths,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Downloaded {len(downloaded)} path(s)", file=sys.stderr)
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
