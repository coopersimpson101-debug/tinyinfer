# Security policy

tinyinfer parses GGUF model files, which are often downloaded from the internet.
The loader is meant to reject malformed files (bounds-checked offsets, sizes,
string lengths and tensor extents) instead of reading out of bounds, so a crash,
hang or out-of-bounds access on a crafted `.gguf` or tokenizer is treated as a
security bug.

## Reporting

Please use GitHub's private vulnerability reporting
(**Security -> Report a vulnerability** on the repository page) rather than a
public issue. Include the file (or a script that generates it), the command
line, and the output. Fixes land on `main`; there are no maintained release branches.
