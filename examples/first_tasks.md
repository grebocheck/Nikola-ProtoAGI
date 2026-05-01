# First ProtoAGI Tasks

Use these after the server starts.

```powershell
$env:PYTHONPATH="src"
python -m protoagi chat --allow-write --prompt "Inspect this repository and summarize its architecture."
```

```powershell
$env:PYTHONPATH="src"
python -m protoagi chat --allow-write --allow-shell --prompt "Run the tests, diagnose any failure, and fix it if needed."
```

```powershell
$env:PYTHONPATH="src"
python -m protoagi chat --prompt "Remember that the preferred runtime profile is 8k context with CpuMoE 4."
```

```powershell
$env:PYTHONPATH="src"
python -m protoagi chat --prompt "What do you remember about the preferred runtime profile?"
```

