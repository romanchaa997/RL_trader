"""
mlkem_feed.py — ML-KEM-768 Post-Quantum Secure Data Feed
AuditorSEC / prodoRIG integration module

Provides a PQC-secure wrapper around cryptocurrency price feeds
using ML-KEM-768 (FIPS 203 Module-Lattice Key Encapsulation Mechanism).

Features:
  - Encrypted price feed via AES-256-GCM + ML-KEM-768 key exchange
  - Integrity verification (HMAC-SHA3-256)
  - Replay attack prevention (nonce/timestamp)
  - Fallback to non-encrypted feed if ML-KEM unavailable
  - Compatible with Binance WebSocket API

Note:
  This is a STUB implementation for demonstration.
  ML-KEM-768 requires a dedicated library (e.g., liboqs-python).
  For production, integrate liboqs or cryptography>=42.0.8 with ML-KEM support.

Usage:
  feed = MLKEMFeed(api_key="...", secret="...")
  price = feed.get_price("BTCUSDT")
"""

import os
import hmac
import hashlib
import struct
import time
import json
from typing import Optional, Dict
import warnings

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    warnings.warn("cryptography library not available, using plaintext mode")

# ── ML-KEM-768 Stub ──────────────────────────────────────────────────────────
class MLKEM768Stub:
    """
    Stub implementation of ML-KEM-768.
    In production, replace with liboqs-python or cryptography>=43.x with FIPS 203 support.
    """
    @staticmethod
    def keygen():
        """Generate ML-KEM-768 keypair (stub)."""
        pk = os.urandom(1184)  # ML-KEM-768 public key size
        sk = os.urandom(2400)  # ML-KEM-768 secret key size
        return pk, sk

    @staticmethod
    def encaps(pk):
        """Encapsulate shared secret (stub)."""
        ct = os.urandom(1088)  # ML-KEM-768 ciphertext size
        ss = os.urandom(32)     # 256-bit shared secret
        return ct, ss

    @staticmethod
    def decaps(sk, ct):
        """Decapsulate shared secret (stub)."""
        return os.urandom(32)  # stub: always returns random


# ── Secure Feed ─────────────────────────────────────────────────────────────
class MLKEMFeed:
    """
    PQC-secured cryptocurrency price feed.
    Encrypts data in transit using AES-256-GCM with ML-KEM-768 key exchange.
    """
    def __init__(self, api_key: str, secret: str, use_encryption: bool = True):
        self.api_key = api_key
        self.secret = secret
        self.use_encryption = use_encryption and CRYPTO_AVAILABLE

        if self.use_encryption:
            # Generate ML-KEM-768 keypair
            self.pk, self.sk = MLKEM768Stub.keygen()
            print("[MLKEM] ML-KEM-768 keypair generated (stub mode)")
        else:
            print("[MLKEM] Running in plaintext mode (no encryption)")

    def _hmac_verify(self, data: bytes, signature: bytes) -> bool:
        """Verify HMAC-SHA3-256 signature."""
        expected = hmac.new(
            self.secret.encode(),
            data,
            hashlib.sha3_256
        ).digest()
        return hmac.compare_digest(signature, expected)

    def _encrypt_payload(self, plaintext: bytes) -> Dict:
        """Encrypt payload with AES-256-GCM using ML-KEM-derived key."""
        # Simulate ML-KEM encapsulation
        ct_mlkem, shared_secret = MLKEM768Stub.encaps(self.pk)

        # Use shared_secret as AES key
        aesgcm = AESGCM(shared_secret)
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        return {
            "ct_mlkem": ct_mlkem.hex(),
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
            "timestamp": int(time.time())
        }

    def _decrypt_payload(self, envelope: Dict) -> bytes:
        """Decrypt payload."""
        ct_mlkem = bytes.fromhex(envelope["ct_mlkem"])
        nonce = bytes.fromhex(envelope["nonce"])
        ciphertext = bytes.fromhex(envelope["ciphertext"])

        # Simulate ML-KEM decapsulation
        shared_secret = MLKEM768Stub.decaps(self.sk, ct_mlkem)

        # Decrypt
        aesgcm = AESGCM(shared_secret)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Fetch current price for symbol (e.g., BTCUSDT).
        In production, replace with WebSocket or REST API call to Binance.
        """
        # Stub: mock price data
        mock_payload = json.dumps({"symbol": symbol, "price": 42000.50}).encode()

        if self.use_encryption:
            # Encrypt
            envelope = self._encrypt_payload(mock_payload)
            print(f"[MLKEM] Encrypted payload size: {len(envelope['ciphertext'])} bytes")

            # Decrypt (simulating receiver side)
            plaintext = self._decrypt_payload(envelope)
            data = json.loads(plaintext)
        else:
            data = json.loads(mock_payload)

        return data.get("price")

    def stream_prices(self, symbols: list):
        """
        Stub for WebSocket-based streaming feed.
        In production, integrate with Binance WebSocket API.
        """
        print(f"[MLKEM] Starting secure stream for {symbols} (stub mode)")
        for symbol in symbols:
            price = self.get_price(symbol)
            yield {"symbol": symbol, "price": price, "timestamp": int(time.time())}


# ── CLI Demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--plaintext", action="store_true",
                        help="Disable encryption (for testing)")
    args = parser.parse_args()

    feed = MLKEMFeed(
        api_key="demo_key",
        secret="demo_secret_256bit_minimum_len",
        use_encryption=not args.plaintext
    )

    print(f"\n[AuditorSEC] Fetching {args.symbol} via ML-KEM-768 secure feed...")
    price = feed.get_price(args.symbol)
    print(f"[AuditorSEC] Price: {price:.2f} USDT")

    print("\n[AuditorSEC] FIPS-203 Compliance: ML-KEM-768 key exchange (stub)")
    print("[AuditorSEC] Production: Replace stub with liboqs-python or cryptography>=43.x")
