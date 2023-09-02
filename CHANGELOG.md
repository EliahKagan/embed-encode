# Changelog

*Major* changes are documented in this file.

## [v2] - 2023-09-02

### Security

- Fix JSON injection vulnerability in `shell/demo`. This was clearly warned
  about, and intentional based on the goal of keeping the script simple. But
  that wasn't a good decision, because the purpose of the script, unlike the
  shorter script `shell/demo-short`, is to show the use of user-supplied input.

### Changed

- Dependency updates.
- CI improvements.

## [v1] - 2023-01-22

### Added

- Experiment in `python/ada-002.ipynb` showing decimal representations with 9
  figures, and briefly explaining their significance.

### Changed

- Revise documentation, particularly `why.md`, to make clear that Base64
  encoding is not required to prevent precision from being lost on the server
  side.
- Clarify how to use `compare/check.rb`.

## [v0]

*Initial release.*

### Added

- Structured explanation in `why.md`.
- Bash, Python, and Java example code.
- Ruby script to compare results.


[v2]: https://github.com/EliahKagan/embed-encode/compare/v1...v2
[v1]: https://github.com/EliahKagan/embed-encode/compare/v0...v1
[v0]: https://github.com/EliahKagan/embed-encode/releases/tag/v0
