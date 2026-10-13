"""Bounded, SSRF-resistant HTTP fetching for user-supplied resource URLs."""

from __future__ import annotations

import ipaddress
import socket
import ssl
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_PAGE_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_DOCUMENT_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_BYTES = DEFAULT_PAGE_MAX_BYTES
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_USER_AGENT = "DexGEM-Paper-Reader/1.0 (+https://reader.dex-gem.ai)"
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeURLError(ValueError):
    """The URL targets a scheme, host, or address the importer must not reach."""


class FetchError(RuntimeError):
    """A bounded public-web fetch failed."""


class NetworkFetchError(FetchError):
    """The remote host could not be reached over the selected transport."""


class HTTPStatusError(FetchError):
    """The remote server returned a non-success HTTP status."""


class ResponseTooLargeError(FetchError):
    """The response exceeded the configured byte limit."""


class TooManyRedirectsError(FetchError):
    """The response exceeded the configured redirect limit."""


@dataclass(frozen=True, slots=True)
class FetchedResponse:
    requested_url: str
    final_url: str
    status: int
    headers: Mapping[str, str]
    body: bytes

    @property
    def content_type(self) -> str:
        return str(self.headers.get("Content-Type") or self.headers.get("content-type") or "").split(";", 1)[0].strip().lower()


Resolver = Callable[..., Iterable[tuple]]


def quote_url_path(url: str) -> str:
    """Percent-encode URL path/query characters without double-encoding escapes."""
    parsed = urlsplit(str(url or "").strip())
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            quote(parsed.path, safe="/%:@!$&'()*+,;=-._~"),
            quote(parsed.query, safe="%!$&'()*+,;=:/?@-._~"),
            parsed.fragment,
        )
    )


def _normalized_host(parsed) -> str:
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError(f"Invalid URL authority: {exc}") from exc
    if not host:
        raise UnsafeURLError("URL must include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("URLs containing credentials are not allowed.")
    if port is not None and not 1 <= port <= 65535:
        raise UnsafeURLError("URL port is invalid.")
    return host.rstrip(".").casefold()


def validate_public_url(url: str, *, resolver: Resolver = socket.getaddrinfo) -> str:
    """Validate HTTP(S) syntax and ensure every currently resolved address is public.

    Resolution is deliberately repeated for each request and redirect. Rejecting the
    whole answer set prevents a mixed public/private DNS response from selecting a
    private address in the underlying HTTP client.
    """

    value = quote_url_path(url)
    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError("Only HTTP and HTTPS URLs are allowed.")
    host = _normalized_host(parsed)
    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeURLError("Localhost URLs are not allowed.")

    try:
        literal = ipaddress.ip_address(host.strip("[]"))
        addresses = {literal}
    except ValueError:
        try:
            answers = resolver(host, parsed.port or (443 if parsed.scheme.casefold() == "https" else 80), type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise UnsafeURLError(f"Hostname did not resolve: {host}") from exc
        addresses = set()
        for answer in answers:
            sockaddr = answer[4]
            if not sockaddr:
                continue
            try:
                addresses.add(ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0]))
            except ValueError:
                continue

    if not addresses:
        raise UnsafeURLError(f"Hostname did not resolve to an IP address: {host}")
    blocked = sorted(str(address) for address in addresses if not address.is_global)
    if blocked:
        raise UnsafeURLError(f"URL resolves to a non-public address: {', '.join(blocked)}")
    return value


def https_upgrade_candidate(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme.casefold() != "http":
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port not in {None, 80}:
        return None
    host = parsed.hostname or ""
    if not host:
        return None
    netloc = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return urlunsplit(("https", netloc, parsed.path, parsed.query, ""))


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, *, resolver: Resolver, max_redirects: int):
        super().__init__()
        self._resolver = resolver
        self._max_redirects = max_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        redirect_count = int(getattr(req, "_paper_reader_redirect_count", 0)) + 1
        if redirect_count > self._max_redirects:
            raise TooManyRedirectsError(f"Too many redirects (maximum {self._max_redirects}).")
        absolute_url = validate_public_url(urljoin(req.full_url, newurl), resolver=self._resolver)
        redirected = super().redirect_request(req, fp, code, msg, headers, absolute_url)
        if redirected is not None:
            redirected._paper_reader_redirect_count = redirect_count
        return redirected


def _read_bounded(response, *, max_bytes: int, chunk_size: int = 128 * 1024) -> bytes:  # noqa: ANN001
    raw_length = response.headers.get("Content-Length") if getattr(response, "headers", None) is not None else None
    if raw_length:
        try:
            if int(raw_length) > max_bytes:
                raise ResponseTooLargeError(f"Response exceeds {max_bytes} bytes.")
        except ValueError:
            pass
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLargeError(f"Response exceeds {max_bytes} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch_once(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout: float,
    max_bytes: int,
    max_redirects: int,
    resolver: Resolver,
) -> FetchedResponse:
    url = validate_public_url(url, resolver=resolver)
    opener = build_opener(_SafeRedirectHandler(resolver=resolver, max_redirects=max_redirects))
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with opener.open(request, timeout=timeout) as response:
            # A custom handler or non-HTTP response must not bypass the redirect check.
            final_url = validate_public_url(str(response.geturl() or url), resolver=resolver)
            body = _read_bounded(response, max_bytes=max_bytes)
            response_headers = {str(key): str(value) for key, value in response.headers.items()}
            return FetchedResponse(
                requested_url=url,
                final_url=final_url,
                status=int(getattr(response, "status", 200) or 200),
                headers=response_headers,
                body=body,
            )
    except (UnsafeURLError, ResponseTooLargeError, TooManyRedirectsError):
        raise
    except HTTPError as exc:
        raise HTTPStatusError(f"HTTP {exc.code} for {url}") from exc
    except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise NetworkFetchError(f"Could not fetch {url}: {reason}") from exc


def fetch_public_url(
    url: str,
    *,
    accept: str = "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_PAGE_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    user_agent: str = DEFAULT_USER_AGENT,
    prefer_https: bool = True,
    resolver: Resolver = socket.getaddrinfo,
) -> FetchedResponse:
    """Fetch a public HTTP(S) resource within explicit redirect/time/size bounds.

    Plain HTTP input is first attempted over HTTPS when it uses the default port.
    If that transport is unavailable, the original validated HTTP URL is fetched.
    """

    headers = {"User-Agent": user_agent, "Accept": accept, "Accept-Encoding": "identity"}
    validated_url = validate_public_url(url, resolver=resolver)
    upgraded = https_upgrade_candidate(validated_url) if prefer_https else None
    if upgraded:
        try:
            return _fetch_once(
                upgraded,
                headers=headers,
                timeout=timeout,
                max_bytes=max_bytes,
                max_redirects=max_redirects,
                resolver=resolver,
            )
        except NetworkFetchError:
            # HTTPS preference is opportunistic only when that transport is unavailable.
            pass
    return _fetch_once(
        validated_url,
        headers=headers,
        timeout=timeout,
        max_bytes=max_bytes,
        max_redirects=max_redirects,
        resolver=resolver,
    )
