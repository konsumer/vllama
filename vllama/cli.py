"""vllama CLI — ollama-compatible interface."""

from __future__ import annotations
import sys
import time
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .config import DAEMON_URL, DAEMON_PORT, DAEMON_HOST

app = typer.Typer(help="vllama — vllm with ollama-style UX", add_completion=False)
console = Console()
err = Console(stderr=True)


def _client() -> httpx.Client:
    return httpx.Client(base_url=DAEMON_URL, timeout=300)


def _daemon_running() -> bool:
    try:
        with _client() as c:
            c.get("/").raise_for_status()
        return True
    except Exception:
        return False


def _require_daemon():
    if not _daemon_running():
        err.print("[red]vllama daemon not running.[/red] Start with: [bold]vllama serve[/bold]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option(DAEMON_HOST, help="Bind host"),
    port: int = typer.Option(DAEMON_PORT, help="Bind port"),
):
    """Start the vllama daemon (foreground)."""
    import uvicorn
    from .daemon import app as daemon_app

    console.print(f"[green]vllama[/green] listening on {host}:{port}")
    uvicorn.run(daemon_app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


@app.command()
def pull(model: str = typer.Argument(..., help="Model name or HF ID")):
    """Download a model from HuggingFace."""
    _require_daemon()
    with _client() as c:
        with c.stream("POST", "/api/pull", json={"name": model}) as resp:
            resp.raise_for_status()
            import json

            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                status = data.get("status", "")
                if data.get("error"):
                    err.print(f"[red]error:[/red] {data['error']}")
                    raise typer.Exit(1)
                console.print(status)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or HF ID"),
    prompt: Optional[str] = typer.Option(None, help="Single prompt (non-interactive)"),
):
    """Run a model interactively (pulls and loads if needed)."""
    _require_daemon()

    # Ensure model is downloaded first
    console.print(f"[dim]Checking {model}...[/dim]")
    with _client() as c:
        resp = c.post("/api/pull", json={"name": model})
        # pull streams; we just fire-and-forget for the check, daemon handles caching

    if prompt:
        _chat_once(model, prompt)
    else:
        _chat_loop(model)


def _chat_once(model: str, prompt: str):
    import json

    with _client() as c:
        with c.stream("POST", "/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                except Exception:
                    pass
    print()


def _chat_loop(model: str):
    import json

    console.print(f"[green]{model}[/green] [dim](type /bye to exit)[/dim]\n")
    history = []

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input in ("/bye", "/exit", "/quit", ""):
            break

        history.append({"role": "user", "content": user_input})

        print("", end="")
        assistant_content = ""

        with _client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": model,
                "messages": history,
                "stream": True,
            }) as resp:
                if resp.status_code != 200:
                    err.print(f"[red]Error {resp.status_code}[/red]")
                    continue
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"].get("content", "")
                        print(delta, end="", flush=True)
                        assistant_content += delta
                    except Exception:
                        pass
        print("\n")
        history.append({"role": "assistant", "content": assistant_content})


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_models():
    """List downloaded models."""
    _require_daemon()
    with _client() as c:
        resp = c.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()

    table = Table(show_header=True)
    table.add_column("NAME", style="cyan")
    table.add_column("ID")
    table.add_column("SIZE", justify="right")
    table.add_column("MODIFIED")

    for m in data.get("models", []):
        size_gb = m["size"] / 1024**3 if m.get("size") else 0
        table.add_row(
            m.get("name", m["model"]),
            m["model"],
            f"{size_gb:.1f} GB",
            str(m.get("modified_at", ""))[:10],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# ps
# ---------------------------------------------------------------------------


@app.command()
def ps():
    """Show running model processes."""
    _require_daemon()
    with _client() as c:
        resp = c.get("/api/ps")
        resp.raise_for_status()
        data = resp.json()

    table = Table(show_header=True)
    table.add_column("NAME", style="cyan")
    table.add_column("ID")
    table.add_column("PID", justify="right")
    table.add_column("PORT", justify="right")
    table.add_column("LOADED")

    for m in data.get("models", []):
        loaded = time.strftime("%H:%M:%S", time.localtime(m.get("loaded_at", 0)))
        table.add_row(
            m.get("name", m["model"]),
            m["model"],
            str(m.get("pid", "")),
            str(m.get("port", "")),
            loaded,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command()
def show(model: str = typer.Argument(..., help="Model name or HF ID")):
    """Show model information."""
    _require_daemon()
    with _client() as c:
        resp = c.post("/api/show", json={"name": model})
        resp.raise_for_status()
        data = resp.json()

    console.print(f"[bold]Model:[/bold]     {data['model']}")
    console.print(f"[bold]Alias:[/bold]     {data.get('alias', '-')}")
    console.print(f"[bold]Pipeline:[/bold]  {data.get('pipeline_tag', '-')}")
    console.print(f"[bold]Downloads:[/bold] {data.get('downloads', '-')}")
    console.print(f"[bold]Likes:[/bold]     {data.get('likes', '-')}")
    if data.get("tags"):
        console.print(f"[bold]Tags:[/bold]      {', '.join(data['tags'][:10])}")


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------


@app.command()
def rm(model: str = typer.Argument(..., help="Model name or HF ID")):
    """Delete a model from disk."""
    _require_daemon()
    typer.confirm(f"Delete {model}?", abort=True)
    with _client() as c:
        resp = c.request("DELETE", "/api/delete", json={"name": model})
        resp.raise_for_status()
    console.print(f"[green]deleted[/green] {model}")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@app.command()
def stop(model: str = typer.Argument(..., help="Model name or HF ID")):
    """Stop a running model (free VRAM)."""
    _require_daemon()
    with _client() as c:
        resp = c.post("/api/stop", json={"name": model})
        resp.raise_for_status()
    console.print(f"[green]stopped[/green] {model}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status():
    """Show memory budget, GPU info, and loaded models."""
    _require_daemon()
    with _client() as c:
        resp = c.get("/api/status")
        resp.raise_for_status()
        data = resp.json()

    console.print(f"\n[bold]Memory[/bold]  {data['used_gb']:.1f} GB used / {data['total_gb']:.1f} GB total  "
                  f"([green]{data['free_gb']:.1f} GB free[/green])\n")

    table = Table(title="GPUs", show_header=True)
    table.add_column("GPU")
    table.add_column("USED", justify="right")
    table.add_column("TOTAL", justify="right")
    for g in data.get("gpus", []):
        table.add_row(g["name"], f"{g['used_gb']:.1f} GB", f"{g['total_gb']:.1f} GB")
    console.print(table)

    running = data.get("models", [])
    if running:
        console.print()
        t2 = Table(title="Loaded models", show_header=True)
        t2.add_column("MODEL")
        t2.add_column("PORT", justify="right")
        t2.add_column("PID", justify="right")
        for m in running:
            t2.add_row(m["model"], str(m["port"]), str(m["pid"]))
        console.print(t2)
    else:
        console.print("[dim]No models currently loaded.[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app()


if __name__ == "__main__":
    main()
