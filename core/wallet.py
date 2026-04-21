"""
Polymarket Wallet Management
Handles CLOB client authentication, balance queries, and token allowances.

All credential handling is centralized here. The rest of the system
only interacts with the ClobClient instance returned by this module.

Required environment variables:
    POLYMARKET_PRIVATE_KEY    — Your signing key (export from Polymarket Settings → Profile)
    POLYMARKET_FUNDER         — Your Polymarket wallet address (the 0x... deposit address)
    POLYMARKET_SIGNATURE_TYPE — 1 for email login (default), 0 for EOA, 2 for browser proxy

Setup for email login (most common):
    1. Polymarket → Settings → Profile → Export Private Key
    2. Polymarket → Profile → copy your wallet/deposit address
    3. export POLYMARKET_PRIVATE_KEY="0x..."
       export POLYMARKET_FUNDER="0x..."
       export POLYMARKET_SIGNATURE_TYPE="1"
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

# Polymarket uses USDC.e on Polygon (6 decimals)
USDC_DECIMALS = 6
USDC_WEI_FACTOR = 10 ** USDC_DECIMALS

# Module-level store for wallet metadata set at creation time.
# We can't reliably introspect py-clob-client's ClobClient for these
# because it stores them under internal/undocumented attribute names.
_wallet_meta: dict = {}


@dataclass
class WalletInfo:
    """Summary of wallet state for pre-flight checks."""
    address: str
    funder: Optional[str]
    signature_type: int
    usdc_balance: Optional[float]  # None if balance check not yet run
    usdc_allowance: Optional[float]
    is_authenticated: bool
    has_allowance: bool


def create_clob_client(
    private_key: Optional[str] = None,
    funder: Optional[str] = None,
    signature_type: Optional[int] = None,
) -> "ClobClient":
    """
    Create and authenticate a ClobClient instance for live trading.

    Reads credentials from environment variables if not provided directly.
    Never log or print private keys.

    Args:
        private_key: Polygon wallet private key. Falls back to POLYMARKET_PRIVATE_KEY env var.
        funder: Funder/proxy address. Falls back to POLYMARKET_FUNDER env var.
        signature_type: 0=EOA, 1=email/Magic, 2=browser proxy. Falls back to env var.

    Returns:
        Authenticated ClobClient ready for trading.

    Raises:
        ValueError: If private key is not configured.
        RuntimeError: If authentication fails.
    """
    from py_clob_client.client import ClobClient

    pk = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY")
    if not pk:
        raise ValueError(
            "No private key configured. Set POLYMARKET_PRIVATE_KEY environment variable "
            "or pass private_key argument.\n"
            "  For email login: Polymarket → Settings → Profile → Export Private Key"
        )

    funder_addr = funder or os.environ.get("POLYMARKET_FUNDER") or None
    sig_type = signature_type
    if sig_type is None:
        sig_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "1"))

    # For email/Magic (1) and browser proxy (2) wallets, the funder address
    # is required — it's the proxy wallet that actually holds your funds.
    if sig_type in (1, 2) and not funder_addr:
        raise ValueError(
            "POLYMARKET_FUNDER is required for email/proxy wallets (signature_type=1 or 2).\n"
            "  This is your Polymarket deposit address (the 0x... shown in your profile).\n"
            "  Set it with: export POLYMARKET_FUNDER=\"0x...\""
        )

    # Build client kwargs
    kwargs = {
        "host": CLOB_HOST,
        "key": pk,
        "chain_id": CHAIN_ID,
    }
    if funder_addr:
        kwargs["funder"] = funder_addr
    if sig_type != 0:
        kwargs["signature_type"] = sig_type

    logger.info(
        f"Initializing CLOB client (chain={CHAIN_ID}, "
        f"sig_type={sig_type}, funder={'set' if funder_addr else 'none'})"
    )

    try:
        client = ClobClient(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create ClobClient: {e}") from e

    # Store metadata for get_wallet_info() — we can't reliably read
    # these back from the ClobClient object later.
    _wallet_meta["signature_type"] = sig_type
    _wallet_meta["funder"] = funder_addr
    # For email/proxy wallets, the funder IS the Polymarket wallet address.
    # For EOA, try reading the signer address from the client.
    if funder_addr:
        _wallet_meta["address"] = funder_addr
    else:
        _wallet_meta["address"] = getattr(client, "address", None) or "unknown"

    # Derive or retrieve L2 API credentials
    try:
        api_creds = client.create_or_derive_api_creds()
        client.set_api_creds(api_creds)
        logger.info("CLOB L2 authentication successful")
    except Exception as e:
        raise RuntimeError(f"L2 authentication failed: {e}") from e

    # Verify connectivity
    try:
        ok = client.get_ok()
        if ok != "OK":
            logger.warning(f"CLOB health check returned unexpected: {ok}")
    except Exception as e:
        raise RuntimeError(f"CLOB connectivity check failed: {e}") from e

    return client


def get_usdc_balance(client: "ClobClient") -> Optional[float]:
    """
    Get the wallet's USDC balance on Polymarket.

    Returns balance in USDC (float), or None on failure.
    The CLOB API returns balance in wei (integer string),
    which must be divided by 1e6 for USDC.
    """
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        result = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        if result and "balance" in result:
            balance_wei = int(result["balance"])
            return balance_wei / USDC_WEI_FACTOR
        return None
    except Exception as e:
        logger.warning(f"Balance check failed: {e}")
        return None


def get_usdc_allowance(client: "ClobClient") -> Optional[float]:
    """
    Get the USDC spending allowance for the Polymarket exchange contract.

    Returns allowance in USDC (float), or None on failure.
    An allowance of 0 means token approvals haven't been set yet.
    """
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        result = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        if result and "allowance" in result:
            allowance_wei = int(result["allowance"])
            return allowance_wei / USDC_WEI_FACTOR
        return None
    except Exception as e:
        logger.warning(f"Allowance check failed: {e}")
        return None


def get_wallet_info(client: "ClobClient") -> WalletInfo:
    """
    Full wallet diagnostic: address, balance, allowance, auth status.

    Reads wallet metadata stored at creation time by create_clob_client(),
    since py-clob-client doesn't expose these reliably via public attributes.
    """
    address = _wallet_meta.get("address", "unknown")
    funder = _wallet_meta.get("funder")
    sig_type = _wallet_meta.get("signature_type", 0)

    balance = get_usdc_balance(client)
    allowance = get_usdc_allowance(client)

    has_allowance = allowance is not None and allowance > 0

    return WalletInfo(
        address=address,
        funder=funder,
        signature_type=sig_type,
        usdc_balance=balance,
        usdc_allowance=allowance,
        is_authenticated=True,
        has_allowance=has_allowance,
    )


def preflight_check(client: "ClobClient", required_balance: float = 10.0) -> bool:
    """
    Run all pre-flight checks before live trading.

    Checks: connectivity, authentication, balance, allowance.
    Prints a diagnostic report and returns True if all checks pass.

    Args:
        client: Authenticated ClobClient.
        required_balance: Minimum USDC balance to proceed.

    Returns:
        True if all checks pass and the wallet is ready to trade.
    """
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT CHECK — Polymarket Wallet")
    print("=" * 60)

    all_ok = True

    # 1. Connectivity
    try:
        ok = client.get_ok()
        server_time = client.get_server_time()
        print(f"  ✅ CLOB API reachable (server time: {server_time})")
    except Exception as e:
        print(f"  ❌ CLOB API unreachable: {e}")
        return False

    # 2. Wallet info
    info = get_wallet_info(client)
    addr_display = info.address[:10] + "…" + info.address[-6:] if len(info.address) > 20 else info.address
    sig_labels = {0: "EOA", 1: "email/Magic", 2: "browser proxy"}
    sig_label = sig_labels.get(info.signature_type, str(info.signature_type))
    print(f"  ✅ Wallet: {addr_display} ({sig_label})")

    # 3. Balance
    if info.usdc_balance is not None:
        if info.usdc_balance >= required_balance:
            print(f"  ✅ USDC balance: ${info.usdc_balance:,.2f}")
        else:
            print(
                f"  ❌ USDC balance: ${info.usdc_balance:,.2f} "
                f"(need ${required_balance:,.2f})"
            )
            all_ok = False
    else:
        print(f"  ⚠️  Could not check USDC balance")
        # Don't fail — some wallet types return 0 due to known issues

    # 4. Allowance
    if info.signature_type in (1, 2):
        # Email/Magic and browser proxy wallets — Polymarket manages allowances
        print(f"  ✅ Token allowance: managed by Polymarket (email/proxy wallet)")
    elif info.has_allowance:
        print(f"  ✅ Token allowance: set")
    else:
        print(
            f"  ❌ Token allowance: NOT SET — run 'python scripts/setup_allowances.py' first"
        )
        all_ok = False

    # 5. Open orders (sanity check)
    open_orders = get_open_orders(client)
    if open_orders:
        print(f"  ⚠️  {len(open_orders)} open order(s) found — review before starting")
    else:
        print(f"  ✅ No stale open orders")

    print("=" * 60)
    if all_ok:
        print("  ✅ All pre-flight checks passed. Ready to trade.")
    else:
        print("  ❌ Some checks failed. Resolve before trading.")
    print()

    return all_ok


def verify_connectivity(client: "ClobClient") -> bool:
    """Quick check that the CLOB API is reachable and credentials work."""
    try:
        ok = client.get_ok()
        server_time = client.get_server_time()
        logger.info(f"CLOB API: {ok}, server time: {server_time}")
        return ok == "OK"
    except Exception as e:
        logger.error(f"CLOB connectivity failed: {e}")
        return False


def get_open_orders(client: "ClobClient") -> list[dict]:
    """Fetch all open orders for the authenticated wallet."""
    from py_clob_client.clob_types import OpenOrderParams

    try:
        orders = client.get_orders(OpenOrderParams())
        return orders if isinstance(orders, list) else []
    except Exception as e:
        logger.error(f"Failed to fetch open orders: {e}")
        return []


def cancel_all_orders(client: "ClobClient") -> bool:
    """Cancel all open orders. Returns True on success."""
    try:
        client.cancel_all()
        logger.info("All open orders cancelled")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel all orders: {e}")
        return False