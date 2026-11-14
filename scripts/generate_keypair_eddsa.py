#!/usr/bin/env python3
"""Generate an Ed25519 key pair for jwt signing.

Creates:
  - keys/auth_token.key     (private key, PEM PKCS8)
  - keys/auth_token.key.pub (public key, PEM)

Usage:
    python generate_keypair_eddsa.py
    python generate_keypair_eddsa.py --out-dir /path/to/keys
"""

import argparse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def get_private_key():
    """
    Returns the private key for signing JWTs.
    """
    keypath = Path("secrets/auth/auth_token.key")
    return serialization.load_pem_private_key(keypath.read_bytes(), password=None)


def make_token():
    import jwt
    from datetime import datetime, timedelta, UTC

    alg = "EdDSA"
    key = get_private_key()
    expiration = datetime.now(UTC) + timedelta(days=30)

    claims = {
        "exp": expiration,
        "iss": "hlgam_auth_server",
        "aud": "hlgam_websocket_client",
    }

    return jwt.encode(claims, key, algorithm=alg)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Ed25519 key pair for JWT signing"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, help="Output directory (default: ../keys/)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("secrets/auth")
    out_dir.mkdir(parents=True, exist_ok=True)

    private_key_path = out_dir / "auth_token.key"
    public_key_path = out_dir / "auth_token.key.pub"

    if private_key_path.exists():
        answer = input(f"{private_key_path} already exists. Overwrite? [y/N] ")
        if answer.lower() != "y":
            print("Aborted.")
            return

    print("Generating Ed25519 key pair...")
    private_key = Ed25519PrivateKey.generate()

    # Write private key (Ed25519 requires PKCS8 format)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_key_path.write_bytes(private_pem)
    print(f"  Private key: {private_key_path}")

    # Write public key
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_key_path.write_bytes(public_pem)
    print(f"  Public key:  {public_key_path}")

    print("\nDone. Keep the private key secret — only the auth server needs it.")
    print("The public key should be provided to auth clients.")


if __name__ == "__main__":
    # main()
    print(make_token())
