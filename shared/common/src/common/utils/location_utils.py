"""Node location resolution via IP geolocation.

Uses ip-api.com as primary provider (free, no key required, 45 req/min).
Falls back to ipinfo.io if the primary fails.

Example::

    from common.utils.location_utils import resolve_node_location

    location = await resolve_node_location()
    if location:
        print(f"Node is in {location.city}, {location.country} ({location.latitude}, {location.longitude})")
"""

from __future__ import annotations

import asyncio

from loguru import logger


async def _try_ip_api() -> dict | None:
    """Resolve location data via ip-api.com (free, no auth, 45 req/min rate limit)."""
    import aiohttp

    url = (
        "http://ip-api.com/json/"
        "?fields=status,message,country,countryCode,regionName,city,lat,lon,timezone,isp,org,query"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.debug(f"ip-api.com returned HTTP {resp.status}")
                    return None
                data = await resp.json(content_type=None)
                if data.get("status") != "success":
                    logger.debug(f"ip-api.com reported failure: {data.get('message')}")
                    return None
                return {
                    "ip": data.get("query"),
                    "country": data.get("country"),
                    "country_code": data.get("countryCode"),
                    "region": data.get("regionName", ""),
                    "city": data.get("city"),
                    "latitude": float(data["lat"]) if data.get("lat") is not None else None,
                    "longitude": float(data["lon"]) if data.get("lon") is not None else None,
                    "timezone": data.get("timezone"),
                    "isp": data.get("isp"),
                    "org": data.get("org"),
                }
    except Exception as exc:
        logger.debug(f"ip-api.com lookup failed: {exc}")
        return None


async def _try_ipinfo() -> dict | None:
    """Resolve location data via ipinfo.io (free tier, 50k req/month, no auth needed)."""
    import aiohttp

    url = "https://ipinfo.io/json"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.debug(f"ipinfo.io returned HTTP {resp.status}")
                    return None
                data = await resp.json(content_type=None)
                loc_str = data.get("loc", "")
                if loc_str and "," in loc_str:
                    lat_str, lon_str = loc_str.split(",", 1)
                    lat: float | None = float(lat_str.strip())
                    lon: float | None = float(lon_str.strip())
                else:
                    lat, lon = None, None
                return {
                    "ip": data.get("ip"),
                    "country": data.get("country"),
                    "country_code": data.get("country"),
                    "region": data.get("region"),
                    "city": data.get("city"),
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": data.get("timezone"),
                    "isp": None,
                    "org": data.get("org"),
                }
    except Exception as exc:
        logger.debug(f"ipinfo.io lookup failed: {exc}")
        return None


async def resolve_node_location(timeout: float = 8.0):
    """Resolve the geographic location of this node by geolocating its public IP.

    Attempts ip-api.com first (free, no key required), then falls back to
    ipinfo.io. Returns ``None`` if all providers fail (e.g. offline, sandboxed,
    or blocked by firewall). This is always best-effort — a failure here must
    never prevent a miner from registering.

    Args:
        timeout: Maximum total seconds to wait across all provider attempts.

    Returns:
        A :class:`~common.models.api_models.NodeLocation` instance on success,
        or ``None`` on failure.
    """
    # Import here to avoid circular imports at module load time
    from common.models.api_models import NodeLocation

    raw: dict | None = None

    try:
        raw = await asyncio.wait_for(_try_ip_api(), timeout=timeout / 2)
    except asyncio.TimeoutError:
        logger.debug("ip-api.com timed out, trying fallback")

    if raw is None:
        try:
            raw = await asyncio.wait_for(_try_ipinfo(), timeout=timeout / 2)
        except asyncio.TimeoutError:
            logger.debug("ipinfo.io timed out")

    if raw is None:
        logger.warning(
            "⚠️  Could not resolve node location — ip-api.com and ipinfo.io are both unreachable. "
            "Location will not be reported for this node."
        )
        return None

    location = NodeLocation(**raw)
    logger.info(
        f"📍 Node location resolved: {location.city}, {location.country} " f"({location.latitude}, {location.longitude})"
    )
    return location
