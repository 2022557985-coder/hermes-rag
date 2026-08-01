"""Web page parser using BeautifulSoup."""

import logging
import re
import time
from typing import Dict, Any
from urllib.parse import urlparse

from .parser_factory import BaseParser
from src.utils.security import validate_url

logger = logging.getLogger("hermes_rag")


class WebParser(BaseParser):
    """Parse web pages using BeautifulSoup with retry support."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def parse(self, source: str) -> Dict[str, Any]:
        """Parse a web page.

        Args:
            source: URL to parse.

        Returns:
            dict with text, tables, metadata.

        Raises:
            ValueError: If the URL is invalid or the response is not usable.
            ConnectionError: If all retries are exhausted.
        """
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        last_exception = None
        for attempt in range(self._max_retries):
            try:
                # SSRF protection: validate URL before fetching
                validate_url(source)

                response = requests.get(source, headers=headers, timeout=15)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or "utf-8"
                break
            except requests.ConnectionError as e:
                last_exception = e
                logger.warning(f"Web fetch attempt {attempt + 1}/{self._max_retries} failed (connection): {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
            except requests.Timeout as e:
                last_exception = e
                logger.warning(f"Web fetch attempt {attempt + 1}/{self._max_retries} timed out: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
            except requests.HTTPError as e:
                logger.error(f"HTTP error fetching {source}: {e}")
                raise ValueError(f"HTTP {e.response.status_code} error for {source}") from e
            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"Web fetch attempt {attempt + 1}/{self._max_retries} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        else:
            raise ConnectionError(f"Failed to fetch {source} after {self._max_retries} attempts") from last_exception
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract title
        title = soup.title.string if soup.title else ""

        # Extract main content
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean text
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Extract tables
        tables = []
        for i, table in enumerate(soup.find_all("table")):
            md_table = self._html_table_to_markdown(table)
            if md_table:
                tables.append({"index": i, "content": md_table})

        parsed_url = urlparse(source)
        metadata = {
            "source": parsed_url.netloc,
            "url": source,
            "type": "web",
            "title": title,
        }

        return {
            "text": text,
            "tables": tables,
            "metadata": metadata,
        }

    def _html_table_to_markdown(self, table) -> str:
        """Convert HTML table to Markdown."""
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        lines = []
        header = "| " + " | ".join(rows[0]) + " |"
        lines.append(header)
        sep = "|" + "|".join("---" for _ in rows[0]) + "|"
        lines.append(sep)
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)