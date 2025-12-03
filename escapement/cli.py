"""
Escapement CLI - Command line interface for managing traces.

Usage:
    escapement list              - List all traces
    escapement show <trace_id>   - Show trace details
    escapement export <trace_id> - Export trace to JSON
    escapement import <file>     - Import trace from JSON
    escapement delete <trace_id> - Delete a trace
    escapement replay <trace_id> - Show replay instructions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

from escapement.config import get_config
from escapement.storage import TraceStorage
from escapement.tracer import StepType

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="escapement")
def main() -> None:
    """Escapement - Deterministic Replay Engine for AI Agents"""
    pass


@main.command()
@click.option("--limit", "-n", default=20, help="Number of traces to show")
@click.option("--status", "-s", type=str, help="Filter by status")
@click.option("--tag", "-t", multiple=True, help="Filter by tag")
def list(limit: int, status: str | None, tag: tuple[str, ...]) -> None:
    """List all recorded traces."""
    config = get_config()
    config.ensure_storage_dir()
    storage = TraceStorage()

    traces = storage.list_traces(
        limit=limit,
        status=status,
        tags=list(tag) if tag else None,
    )

    if not traces:
        console.print("[dim]No traces found.[/dim]")
        console.print(f"[dim]Storage location: {config.get_db_path()}[/dim]")
        return

    table = Table(
        title="Recorded Traces",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Trace ID", style="green")
    table.add_column("Name", style="white")
    table.add_column("Status", style="yellow")
    table.add_column("Steps", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Created", style="dim")

    for trace in traces:
        status_style = {
            "completed": "green",
            "failed": "red",
            "failed_loop": "red bold",
            "running": "yellow",
            "interrupted": "orange3",
        }.get(trace.status, "white")

        table.add_row(
            trace.trace_id,
            trace.name or "-",
            f"[{status_style}]{trace.status}[/{status_style}]",
            str(trace.total_steps),
            str(trace.total_tokens),
            trace.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(traces)} of {limit} max traces[/dim]")


@main.command()
@click.argument("trace_id")
@click.option("--steps/--no-steps", default=True, help="Show step details")
@click.option("--content/--no-content", default=False, help="Show message content")
def show(trace_id: str, steps: bool, content: bool) -> None:
    """Show detailed information about a trace."""
    storage = TraceStorage()
    trace = storage.load_trace(trace_id)

    if trace is None:
        console.print(f"[red]Trace not found: {trace_id}[/red]")
        sys.exit(1)

    # Header panel
    header = f"""[bold cyan]Trace:[/bold cyan] {trace.trace_id}
[bold]Name:[/bold] {trace.metadata.name or 'unnamed'}
[bold]Status:[/bold] {trace.metadata.status}
[bold]Created:[/bold] {trace.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}
[bold]Steps:[/bold] {trace.metadata.total_steps}
[bold]Tokens:[/bold] {trace.metadata.total_tokens}
[bold]Tags:[/bold] {', '.join(trace.metadata.tags) or 'none'}"""

    console.print(Panel(header, title="Trace Overview", border_style="cyan"))

    if not steps:
        return

    # Build step tree
    tree = Tree(f"[bold]Execution Steps ({len(trace.steps)})[/bold]")

    for step in trace.steps:
        step_icon = {
            StepType.LLM_CALL: "[cyan]LLM[/cyan]",
            StepType.TOOL_CALL: "[yellow]TOOL[/yellow]",
            StepType.USER_INPUT: "[green]USER[/green]",
            StepType.AGENT_OUTPUT: "[blue]OUT[/blue]",
            StepType.ERROR: "[red]ERR[/red]",
            StepType.CHECKPOINT: "[magenta]CHK[/magenta]",
        }.get(step.step_type, "[dim]???[/dim]")

        step_label = f"{step_icon} Step {step.step_number}"

        if step.step_type == StepType.LLM_CALL and step.llm_request:
            model = step.llm_request.model
            tokens = ""
            if step.llm_response and step.llm_response.usage:
                tokens = f" ({step.llm_response.usage.get('total_tokens', 0)} tokens)"
            step_label += f" - {model}{tokens}"

            if step.duration_ms:
                step_label += f" [{step.duration_ms:.0f}ms]"

        elif step.step_type == StepType.TOOL_CALL and step.tool_request:
            step_label += f" - {step.tool_request.tool_name}"

        elif step.step_type == StepType.CHECKPOINT:
            step_label += f" - {step.data.get('name', 'checkpoint')}"

        branch = tree.add(step_label)

        # Show content if requested
        if content and step.step_type == StepType.LLM_CALL:
            if step.llm_request:
                for msg in step.llm_request.messages[-2:]:  # Last 2 messages
                    role = msg.get("role", "?")
                    msg_content = msg.get("content", "")
                    if isinstance(msg_content, str):
                        preview = msg_content[:100] + "..." if len(msg_content) > 100 else msg_content
                        branch.add(f"[dim]{role}:[/dim] {preview}")

            if step.llm_response and step.llm_response.choices:
                choice = step.llm_response.choices[0]
                assistant_content = choice.get("message", {}).get("content", "")
                if assistant_content:
                    preview = (
                        assistant_content[:100] + "..."
                        if len(assistant_content) > 100
                        else assistant_content
                    )
                    branch.add(f"[dim]assistant:[/dim] {preview}")

    console.print(tree)

    # Replay instructions
    console.print("\n[bold]To replay this trace:[/bold]")
    console.print(f"""
[dim]import escapement
escapement.replay("{trace_id}")
# Your agent code here...[/dim]
""")


@main.command()
@click.argument("trace_id")
@click.argument("output", type=click.Path(), required=False)
def export(trace_id: str, output: str | None) -> None:
    """Export a trace to JSON file."""
    storage = TraceStorage()
    trace = storage.load_trace(trace_id)

    if trace is None:
        console.print(f"[red]Trace not found: {trace_id}[/red]")
        sys.exit(1)

    output_path = Path(output) if output else Path(f"{trace_id}.json")

    trace_dict = trace.to_dict()
    output_path.write_text(json.dumps(trace_dict, indent=2, default=str))

    console.print(f"[green]Exported trace to {output_path}[/green]")
    console.print(f"[dim]Size: {output_path.stat().st_size / 1024:.1f} KB[/dim]")


@main.command(name="import")
@click.argument("input_file", type=click.Path(exists=True))
def import_trace(input_file: str) -> None:
    """Import a trace from JSON file."""
    storage = TraceStorage()
    input_path = Path(input_file)

    trace = storage.import_trace(input_path)

    if trace is None:
        console.print(f"[red]Failed to import trace from {input_file}[/red]")
        sys.exit(1)

    console.print(f"[green]Imported trace: {trace.trace_id}[/green]")
    console.print(f"[dim]Steps: {trace.metadata.total_steps}[/dim]")


@main.command()
@click.argument("trace_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def delete(trace_id: str, force: bool) -> None:
    """Delete a trace."""
    storage = TraceStorage()

    if not force:
        if not click.confirm(f"Delete trace {trace_id}?"):
            return

    if storage.delete_trace(trace_id):
        console.print(f"[green]Deleted trace: {trace_id}[/green]")
    else:
        console.print(f"[red]Trace not found: {trace_id}[/red]")
        sys.exit(1)


@main.command()
@click.argument("trace_id")
def stats(trace_id: str) -> None:
    """Show statistics for a trace."""
    storage = TraceStorage()
    trace = storage.load_trace(trace_id)

    if trace is None:
        console.print(f"[red]Trace not found: {trace_id}[/red]")
        sys.exit(1)

    llm_steps = trace.get_llm_steps()
    tool_steps = trace.get_tool_steps()

    # Model usage breakdown
    model_usage: dict[str, dict[str, Any]] = {}
    for step in llm_steps:
        if step.llm_request:
            model = step.llm_request.model
            if model not in model_usage:
                model_usage[model] = {"calls": 0, "tokens": 0, "duration_ms": 0}
            model_usage[model]["calls"] += 1
            if step.llm_response and step.llm_response.usage:
                model_usage[model]["tokens"] += step.llm_response.usage.get("total_tokens", 0)
            if step.duration_ms:
                model_usage[model]["duration_ms"] += step.duration_ms

    # Tool usage breakdown
    tool_usage: dict[str, int] = {}
    for step in tool_steps:
        if step.tool_request:
            tool = step.tool_request.tool_name
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

    # Print stats
    console.print(Panel(f"[bold]Statistics for {trace_id}[/bold]", border_style="cyan"))

    console.print("\n[bold cyan]LLM Usage by Model:[/bold cyan]")
    model_table = Table(box=box.SIMPLE)
    model_table.add_column("Model")
    model_table.add_column("Calls", justify="right")
    model_table.add_column("Tokens", justify="right")
    model_table.add_column("Avg Latency", justify="right")

    for model, data in model_usage.items():
        avg_latency = data["duration_ms"] / data["calls"] if data["calls"] > 0 else 0
        model_table.add_row(
            model,
            str(data["calls"]),
            str(data["tokens"]),
            f"{avg_latency:.0f}ms",
        )
    console.print(model_table)

    if tool_usage:
        console.print("\n[bold cyan]Tool Usage:[/bold cyan]")
        tool_table = Table(box=box.SIMPLE)
        tool_table.add_column("Tool")
        tool_table.add_column("Calls", justify="right")
        for tool, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
            tool_table.add_row(tool, str(count))
        console.print(tool_table)


@main.command()
def info() -> None:
    """Show Escapement configuration and status."""
    config = get_config()

    console.print(Panel("[bold]Escapement Configuration[/bold]", border_style="cyan"))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Storage Directory", str(config.storage_dir))
    table.add_row("Database", str(config.get_db_path()))
    table.add_row("Recording Enabled", str(config.enabled))
    table.add_row("Loop Detection", str(config.loop_detection_enabled))
    table.add_row("Max Identical Calls", str(config.max_identical_calls))
    table.add_row("Strict Replay", str(config.strict_replay))

    console.print(table)

    # Check if database exists
    db_path = config.get_db_path()
    if db_path.exists():
        storage = TraceStorage()
        traces = storage.list_traces(limit=1000)
        console.print(f"\n[dim]Total traces stored: {len(traces)}[/dim]")
    else:
        console.print("\n[dim]No traces recorded yet.[/dim]")


if __name__ == "__main__":
    main()
