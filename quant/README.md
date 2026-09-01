# Beyond packed-cache runtime

This package provides the packed 4-bit/group-32 cache representation, fused
cache writer, readback utilities, and the production SM103 non-uniform decode
kernel used by the Beyond vLLM backend.

Install it after the root runtime dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ./quant
python -c "import quant.beyond_cute; print('packed runtime import: ok')"
```

The runtime has one production decode-kernel family. Unsupported devices and
layouts fail closed; eager readback is reserved for explicit numerical checks.

Original Beyond code is Apache-2.0. The `fa4_cute/` subtree retains its BSD
3-Clause license and copyright notices; see `fa4_cute/LICENSE` and
`fa4_cute/AUTHORS`.
