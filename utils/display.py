import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class AuraConsoleUI:

    @staticmethod
    def print_banner():
        """Renders the high-contrast strategic initialization header."""
        banner_text = (
            "[bold cyan]AURA: Autonomous Vector-Space Auditing Framework[/bold cyan]\n"
            "[bold black]Local Privacy Evaluation Loop & Embedding Inversion Simulator[/bold black]"
        )
        console.print(
            Panel(banner_text, border_style="cyan", expand=False, padding=(1, 2))
        )
        console.print(
            "[dim cyan][INFO] Active Security Standard Compliance: OWASP LLM08 (Embedding Weaknesses)[/dim cyan]\n"
        )

    @staticmethod
    def render_matrix_row(step, loss, similarity, words):
        """Prints a styled log tracking gradient progression indicators."""
        word_chain = " ".join([f"[green]{w}[/green]" for w in words])
        console.print(
            f" [bold yellow]Step {step:03d}[/bold yellow] │ "
            f"Loss: [bold magenta]{loss:.4f}[/bold magenta] │ "
            f"Cos Sim: [bold cyan]{similarity:.4f}[/bold cyan] │ "
            f"Current Target Vector Estimate: {word_chain}"
        )

    @staticmethod
    def render_summary_report(metrics: dict):
        """Generates a scannable grid showcasing metrics evaluations."""
        table = Table(
            title="\n[bold white]VULNERABILITY ASSESSMENT REPORT[/bold white]",
            title_justify="left",
            border_style="dim cyan",  # Changed from "dim gray" to "dim cyan" to maintain cyberpunk aesthetic safely
        )
        table.add_column("Metric Field", style="cyan")
        table.add_column("Evaluation Trace", style="magenta")
        table.add_column("Risk Classification Status", style="bold red")

        for k, v in metrics.items():
            status = "[bold green]SECURE[/bold green]"
            if "CRITICAL" in str(v["status"]):
                status = "[bold red]CRITICAL LEAK[/bold red]"
            elif "HIGH" in str(v["status"]):
                status = "[bold yellow]HIGH RISK[/bold yellow]"

            table.add_row(k, str(v["value"]), status)

        console.print(table)


if __name__ == "__main__":
    AuraConsoleUI.print_banner()