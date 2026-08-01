# Security Policy

## Reporting a vulnerability

Please do not open a public issue. Report vulnerabilities privately by emailing
the maintainers listed on the GitHub repository page. Include a minimal
reproduction and the affected version.

## Supported versions

| Version | Supported |
|---------|-----------|
| 2.x     | Yes       |
| < 2.0   | No        |

## Security notes

- The API is unauthenticated by default. Set `HERMES_API_KEY` before exposing
  the server outside localhost.
- Ingestion validates file paths, extensions, and file sizes. Do not ingest
  untrusted URLs unless you have reviewed the URL policy.
- The Gradio UI is intended for trusted local users. Do not expose it directly
  to the public internet.