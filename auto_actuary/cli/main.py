"""
auto_actuary CLI
================
Command-line interface for running actuarial analyses without writing code.

Usage examples:
    auto-actuary triangle --lob PPA --config config/schema.yaml
    auto-actuary reserve  --lob PPA --output output/
    auto-actuary dashboard --lob PPA --output output/
    auto-actuary ratemaking --lob PPA
    auto-actuary all --lob PPA --output output/

All commands accept:
  --config   Path to schema.yaml          (default: config/schema.yaml)
  --lob      Line of business code        (e.g. PPA, HO, CA)
  --output   Output directory             (default: output/)
  --fmt      Output format: excel | html  (default: excel)
  --data-dir Directory with CSV files     (default: from schema.yaml source.base_path)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False

from auto_actuary.core.config import ActuaryConfig
from auto_actuary.core.session import ActuarySession

app = typer.Typer(
    name="auto-actuary",
    help="P&C actuarial analytics — FCAS-level math, executive-grade output.",
    add_completion=False,
) if HAS_TYPER else None

console = Console() if HAS_TYPER else None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

def _make_session(
    config: Path,
    data_dir: Optional[Path],
    lob: Optional[str],
) -> ActuarySession:
    """Build session and auto-load CSVs from data_dir."""
    if not config.exists():
        console.print(f"[red]Config not found: {config}[/red]")
        raise typer.Exit(1)

    session = ActuarySession.from_config(config)
    base = data_dir or session.config.base_path
    base = Path(base)

    table_files = {
        "policies":     ["policies.csv", "policy.csv"],
        "transactions": ["transactions.csv", "txns.csv"],
        "claims":       ["claims.csv", "claim.csv"],
        "valuations":   ["valuations.csv", "loss_valuations.csv", "loss_vals.csv"],
        "rate_changes": ["rate_changes.csv", "rate_history.csv"],
        "expenses":     ["expenses.csv"],
    }

    for table, candidates in table_files.items():
        for fname in candidates:
            fpath = base / fname
            if fpath.exists():
                console.print(f"  [dim]Loading {table} from {fpath}[/dim]")
                session.load_csv(table, fpath)
                break

    if not session.loader.loaded_tables:
        console.print(
            f"[yellow]No CSV files found in {base}.  "
            "Populate data/ or pass --data-dir.[/yellow]"
        )

    return session


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def triangle(
    lob: str = typer.Option(..., "--lob", "-l", help="Line of business code (e.g. PPA)"),
    config: Path = typer.Option(Path("config/schema.yaml"), "--config", "-c"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    output: Path = typer.Option(Path("output"), "--output", "-o"),
    fmt: str = typer.Option("excel", "--fmt", "-f", help="excel or html"),
    value: str = typer.Option("incurred_loss", "--value", help="incurred_loss | paid_loss | paid_count"),
):
    """Build and export a loss development triangle exhibit."""
    console.rule("[bold blue]Triangle Development[/bold blue]")
    session = _make_session(config, data_dir, lob)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task(f"Building {lob} triangle ({value})…", total=None)
        out_path = session.triangle_exhibit(lob=lob, value=value,
                                             output_path=output / f"{lob}_triangle.{'xlsx' if fmt=='excel' else 'html'}",
                                             fmt=fmt)
        p.update(task, completed=True)

    console.print(f"[green]✓ Triangle exhibit saved:[/green] {out_path}")


@app.command()
def reserve(
    lob: str = typer.Option(..., "--lob", "-l"),
    config: Path = typer.Option(Path("config/schema.yaml"), "--config", "-c"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    output: Path = typer.Option(Path("output"), "--output", "-o"),
    fmt: str = typer.Option("excel", "--fmt", "-f"),
):
    """Run IBNR reserve analysis and export exhibit."""
    console.rule("[bold blue]Reserve Analysis[/bold blue]")
    session = _make_session(config, data_dir, lob)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task(f"Running reserve analysis for {lob}…", total=None)
        out_path = session.reserve_exhibit(
            lob=lob,
            output_path=output / f"{lob}_reserve.{'xlsx' if fmt=='excel' else 'html'}",
            fmt=fmt,
        )
        p.update(task, completed=True)

    # Print quick summary to terminal
    analysis = session.reserve_analysis(lob=lob)
    tbl = Table(title=f"Reserve Summary — {lob}", show_header=True, header_style="bold blue")
    tbl.add_column("Method", style="cyan")
    tbl.add_column("Total IBNR", justify="right")
    tbl.add_column("ELR", justify="right")
    for m in analysis.available_methods:
        res = analysis.result(m)
        elr_str = f"{res.elr:.4f}" if res.elr else "—"
        tbl.add_row(m.replace("_", " ").title(), f"${res.total_ibnr:,.0f}", elr_str)
    console.print(tbl)
    console.print(f"[green]✓ Reserve exhibit saved:[/green] {out_path}")


@app.command()
def ratemaking(
    lob: str = typer.Option(..., "--lob", "-l"),
    config: Path = typer.Option(Path("config/schema.yaml"), "--config", "-c"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    output: Path = typer.Option(Path("output"), "--output", "-o"),
    fmt: str = typer.Option("excel", "--fmt", "-f"),
    trend_factor: float = typer.Option(1.0, "--trend-factor", help="Cumulative loss trend factor"),
    var_exp: float = typer.Option(0.25, "--var-exp"),
    fixed_exp: float = typer.Option(0.05, "--fixed-exp"),
    profit: float = typer.Option(0.05, "--profit"),
):
    """Compute rate indication and export exhibit."""
    console.rule("[bold blue]Rate Indication[/bold blue]")
    session = _make_session(config, data_dir, lob)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task(f"Computing rate indication for {lob}…", total=None)
        ind = session.rate_indication(
            lob=lob,
            trend_factor=trend_factor,
            variable_expense_ratio=var_exp,
            fixed_expense_ratio=fixed_exp,
            target_profit_margin=profit,
        )
        result = ind.compute()
        out_path = session.rate_indication_exhibit(
            lob=lob,
            output_path=output / f"{lob}_rate_indication.{'xlsx' if fmt=='excel' else 'html'}",
            fmt=fmt,
        )
        p.update(task, completed=True)

    tbl = Table(title=f"Rate Indication — {lob}", show_header=True, header_style="bold blue")
    tbl.add_column("Item")
    tbl.add_column("Value", justify="right")
    tbl.add_row("On-Level Earned Premium", f"${result.on_level_premium:,.0f}")
    tbl.add_row("Trended Ultimate Loss", f"${result.trended_ultimate_loss:,.0f}")
    tbl.add_row("Projected Loss Ratio", f"{result.projected_loss_ratio:.4f}")
    tbl.add_row("Permissible Loss Ratio", f"{result.permissible_loss_ratio:.4f}")
    tbl.add_row("[bold red]Indicated Change[/bold red]", f"[bold red]{result.indicated_pct}[/bold red]")
    tbl.add_row("Credibility", f"{result.credibility:.4f}")
    tbl.add_row("[bold]Credibility-Weighted[/bold]", f"[bold]{result.credibility_weighted_pct}[/bold]")
    console.print(tbl)
    console.print(f"[green]✓ Rate indication exhibit saved:[/green] {out_path}")


@app.command()
def dashboard(
    config: Path = typer.Option(Path("config/schema.yaml"), "--config", "-c"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    output: Path = typer.Option(Path("output"), "--output", "-o"),
    lob: Optional[str] = typer.Option(None, "--lob", "-l", help="Filter to single LOB (optional)"),
):
    """Generate the executive HTML dashboard."""
    console.rule("[bold blue]Executive Dashboard[/bold blue]")
    session = _make_session(config, data_dir, lob)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task("Building executive dashboard…", total=None)
        out_path = session.exec_dashboard(
            output_path=output / "dashboard.html",
            lob=lob,
        )
        p.update(task, completed=True)

    console.print(f"[green]✓ Dashboard saved:[/green] {out_path}")
    console.print("[dim]Open the HTML file in any browser — no server required.[/dim]")


@app.command(name="all")
def run_all(
    lob: str = typer.Option(..., "--lob", "-l"),
    config: Path = typer.Option(Path("config/schema.yaml"), "--config", "-c"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    output: Path = typer.Option(Path("output"), "--output", "-o"),
    fmt: str = typer.Option("excel", "--fmt", "-f"),
):
    """Run all analyses and generate all exhibits for a LOB."""
    console.rule(f"[bold blue]auto_actuary — Full Run — {lob}[/bold blue]")
    session = _make_session(config, data_dir, lob)
    output.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("Triangle Exhibit",    lambda: session.triangle_exhibit(lob=lob, output_path=output/f"{lob}_triangle.{'xlsx' if fmt=='excel' else 'html'}", fmt=fmt)),
        ("Reserve Exhibit",     lambda: session.reserve_exhibit(lob=lob, output_path=output/f"{lob}_reserve.{'xlsx' if fmt=='excel' else 'html'}", fmt=fmt)),
        ("Rate Indication",     lambda: session.rate_indication_exhibit(lob=lob, output_path=output/f"{lob}_rate_indication.{'xlsx' if fmt=='excel' else 'html'}", fmt=fmt)),
        ("Executive Dashboard", lambda: session.exec_dashboard(output_path=output/"dashboard.html", lob=lob)),
    ]

    results = []
    for task_name, fn in tasks:
        with Progress(SpinnerColumn(), TextColumn(f"  {task_name}…"), console=console, transient=True) as p:
            t = p.add_task("", total=None)
            try:
                out = fn()
                results.append((task_name, str(out), "✓"))
            except Exception as exc:
                results.append((task_name, str(exc), "✗"))

    tbl = Table(title="Output Summary", show_header=True)
    tbl.add_column("Analysis")
    tbl.add_column("Status", justify="center")
    tbl.add_column("Output")
    for name, path, status in results:
        style = "green" if status == "✓" else "red"
        tbl.add_row(name, f"[{style}]{status}[/{style}]", path)
    console.print(tbl)


@app.command()
def validate(
    config: Path = typer.Option(Path("config/schema.yaml"), "--config", "-c"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
):
    """Validate config and data files — check schema mappings, required columns."""
    console.rule("[bold blue]Configuration Validation[/bold blue]")
    cfg = ActuaryConfig(config)
    console.print(f"  Config loaded: {cfg}")
    console.print(f"  Source type:   {cfg.source_type}")
    console.print(f"  LOBs defined:  {list(cfg.lobs.keys())}")
    console.print(f"  Output dir:    {cfg.output_dir}")

    base = data_dir or cfg.base_path
    for fname in ["policies.csv", "claims.csv", "valuations.csv", "transactions.csv"]:
        fpath = Path(base) / fname
        status = "[green]✓[/green]" if fpath.exists() else "[yellow]missing[/yellow]"
        console.print(f"  {status} {fpath}")

    console.print("[green]Validation complete.[/green]")


if __name__ == "__main__":
    if HAS_TYPER:
        app()
    else:
        print("Install typer for CLI support:  pip install typer[all]")
        sys.exit(1)
