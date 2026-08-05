# Biznes PAP fixtures

`observed_2023_03_16.ajax.json` is a minimized verbatim subset of the official
indexed Drupal Ajax response at `/articles/periodic/2023/3/16`, retaining the
day header and CreativeForge Games SA row. It verifies only the neutral generic
table fields: company ID, report number, date, and detail path.

This fixture does not prove EBI/ESPI channel provenance and must not be used to
construct `DocumentRef` records. NewConnect discovery remains fail-closed behind
PAP's WAF pending a verified browser-capable channel/detail transport.
