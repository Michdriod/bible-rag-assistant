# Import necessary modules and libraries for CLI
import requests
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

# Define functions to send queries and fetch suggestions
BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/bible")
console = Console()


def ask_llm_query(query: str, version: str = "kjv"):
    payload = {"query": query, "include_context": True, "version": version}
    try:
        response = requests.post(f"{BASE_URL}/search", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        console.print(f"[red]❌ Request failed:[/red] {e}")
        return None


def get_suggestions(partial_query: str):
    """Get suggestions for partial queries"""
    try:
        response = requests.get(f"{BASE_URL}/suggestions", params={"q": partial_query})
        response.raise_for_status()
        return response.json().get("suggestions", [])
    except:
        return []


def validate_range(query: str):
    """Validate range syntax before searching"""
    try:
        response = requests.get(f"{BASE_URL}/validate-range", params={"range_query": query})
        response.raise_for_status()
        return response.json()
    except:
        return None


def render_results(response: dict):
    # CASE: Backend classified it as invalid
    if response.get("query_type") == "invalid":
        console.print("[bold red]⚠ This does not appear to be a Bible-related query.[/bold red]")
        return

    # SHOW: Query Type with enhanced descriptions
    query_type = response.get("query_type", "").lower()
    classification = response.get("classification", {})
    version = response.get("version", "kjv").upper()
    
    # Display Bible version used
    console.print(f"[bold cyan]📚 Bible Version: {version}[/bold cyan]")
    
    if query_type == "exact_reference":
        console.print("[bold green]📌 Match Type: Exact Reference[/bold green]")
    elif query_type == "exact_range":
        console.print("[bold green]📌 Match Type: Exact Range[/bold green]")
    elif query_type == "semantic_search":
        console.print("[bold yellow]📌 Match Type: Semantic Search[/bold yellow]")
    else:
        console.print(f"[bold red]📌 Match Type: {query_type.title()}[/bold red]")

    # SHOW: Query with confidence if available
    query_text = response.get('query', 'N/A')
    confidence = classification.get('confidence', 0)
    if confidence > 0:
        console.print(f"[blue]🔍 Query:[/blue] {query_text} [dim](confidence: {confidence:.1%})[/dim]\n")
    else:
        console.print(f"[blue]🔍 Query:[/blue] {query_text}\n")

    # SHOW: Range Information (NEW)
    range_info = response.get("range_info")
    if range_info and range_info.get("is_range"):
        total_requested = range_info.get("total_verses_requested", 0)
        total_found = range_info.get("total_verses_found", 0)
        missing_verses = range_info.get("missing_verses", [])
        
        range_status = Text()
        range_status.append("📊 Range Info: ", style="bold blue")
        range_status.append(f"{total_found}/{total_requested} verses found", style="green")
        
        if missing_verses:
            range_status.append(f" ({len(missing_verses)} missing)", style="yellow")
        
        console.print(range_status)
        
        if missing_verses:
            console.print(f"[dim]Missing: {', '.join(missing_verses[:3])}{'...' if len(missing_verses) > 3 else ''}[/dim]")
        console.print()

    # AI Response Panel
    ai_resp = response.get("ai_response", "")
    if ai_resp and ai_resp != response.get("message", ""):
        console.rule("[bold blue]🧠 AI Response")
        console.print(Panel.fit(ai_resp, border_style="cyan"))
        console.print()
    
    # System Message (if different from AI response)
    message = response.get("message", "")
    if message and message != ai_resp:
        console.print(f"[dim]ℹ️  {message}[/dim]\n")

    # Bible Verses
    verses = response.get("results", [])
    if verses:
        # Enhanced table with better formatting for ranges
        table = Table(title="📖 Bible Verses", show_lines=True, expand=True)
        table.add_column("Reference", style="cyan", no_wrap=False, min_width=12)
        table.add_column("Text", style="white", ratio=3, overflow="fold") #ratio=3
        

        for verse in verses:
            # Truncate very long verses for better display
            text = verse["text"]
            if len(text) > 500:
                text = text[:147] + "..."
            table.add_row(verse["reference"], text)

        console.print(table)
        
    # verses = response.get("results", [])
    # if verses:
    #     table = Table(title="📖 Bible Verses", show_lines=True, expand=True)
    #     table.add_column("Reference", style="cyan", no_wrap=False, min_width=12)
    #     table.add_column("Text", style="white", ratio=3, overflow="fold")  # allow rich to wrap long text

    # for verse in verses:
    #     text = verse["text"]  # Don't truncate manually
    #     table.add_row(verse["reference"], text)

    #     console.print(table)
        
        # Show total count for ranges
        if len(verses) > 1:
            console.print(f"[dim]Total: {len(verses)} verses[/dim]")
    else:
        console.print("[yellow]⚠️ No verses found in database.[/yellow]")

    # Suggestions (Enhanced with better formatting)
    suggestions = response.get("suggestions", [])
    if suggestions:
        console.print("\n[blue]💡 Try these suggestions:[/blue]")
        
        # Group suggestions by type
        refs = [s for s in suggestions if any(c.isdigit() for c in s)]
        topics = [s for s in suggestions if s not in refs]
        
        suggestion_items = []
        for s in suggestions[:6]:  # Limit to 6 suggestions
            suggestion_items.append(f"[cyan]•[/cyan] {s}")
        
        if len(suggestion_items) <= 3:
            for item in suggestion_items:
                console.print(f"  {item}")
        else:
            # Use columns for better layout
            console.print(Columns(suggestion_items, equal=True, expand=True))


def show_examples():
    """Show usage examples"""
    console.print("\n[bold blue]📚 Usage Examples:[/bold blue]")
    
    examples = [
        ("Single Verse", "John 3:16"),
        ("Verse Range", "Genesis 1:1-3"),
        ("Chapter Range", "Psalm 23:1-6"),
        ("Topic Search", "verses about love"),
        ("Quote Search", "For God so loved the world"),
        ("Theme Search", "faith and hope")
    ]
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="green", no_wrap=True)
    table.add_column(style="white")
    # wraps instead of cutting off
    
    for example_type, example_query in examples:
        table.add_row(f"{example_type}:", f'"{example_query}"')
    
    console.print(table)


def show_help():
    """Show help information"""
    help_text = """
[bold blue]🔍 Bible RAG Assistant Help[/bold blue]

[bold yellow]Query Types:[/bold yellow]
• [green]Exact References[/green]: John 3:16, Romans 8:28, 1 John 4:7
• [green]Verse Ranges[/green]: Genesis 1:1-3, Psalm 23:1-6, Matthew 5:3-12
• [green]Semantic Search[/green]: "verses about love", "faith and hope"
• [green]Quote Search[/green]: "For God so loved the world"

[bold yellow]Commands:[/bold yellow]
• [cyan]help[/cyan] - Show this help
• [cyan]examples[/cyan] - Show usage examples
• [cyan]version <version>[/cyan] - Change Bible version (kjv, niv, nkjv, nlt)
• [cyan]validate <range>[/cyan] - Check if a verse range is valid
• [cyan]exit/quit[/cyan] - Exit the program

[bold yellow]Tips:[/bold yellow]
• Use standard Bible book names and abbreviations
• Ranges work within single chapters (e.g., Genesis 1:1-5)
• Be specific with topic searches for better results
• Quotes don't need exact wording - semantic search will find similar verses
"""
    console.print(Panel(help_text, border_style="blue"))


def run_llm_cli():
    console.print("[bold magenta]🧬 Bible RAG Assistant[/bold magenta]")
    console.print("Enhanced with verse range support and multiple Bible versions\n")
    console.print("[dim]Type 'help' for usage examples, 'exit' to quit[/dim]\n")
    
    # Default version
    current_version = "kjv"
    console.print(f"[bold cyan]Current Bible version:[/bold cyan] [green]{current_version.upper()}[/green]")
    
    while True:
        try:
            query = console.input("[bold yellow]🔍 Query[/bold yellow]: ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                console.print("\n👋 Goodbye!")
                break
            elif query.lower() in ["help", "h", "?"]:
                show_help()
                continue
            elif query.lower() in ["examples", "ex"]:
                show_examples()
                continue
            elif query.lower().startswith("validate "):
                # Range validation feature
                range_query = query[9:].strip()
                validation = validate_range(range_query)
                if validation:
                    if validation["valid"]:
                        console.print(f"[green]✅ Valid range:[/green] {range_query}")
                        if validation.get("range_info"):
                            info = validation["range_info"]
                            console.print(f"[dim]Will search for {info.get('total_verses', 'unknown')} verses[/dim]")
                    else:
                        console.print(f"[red]❌ Invalid range:[/red] {validation.get('message', 'Unknown error')}")
                else:
                    console.print("[red]❌ Could not validate range[/red]")
                console.print()
                continue
            elif query.lower().startswith("version "):
                # Change Bible version
                new_version = query[8:].strip().lower()
                supported_versions = ["kjv", "niv", "nkjv", "nlt"]
                if new_version in supported_versions:
                    current_version = new_version
                    console.print(f"[bold green]✅ Bible version changed to:[/bold green] [cyan]{current_version.upper()}[/cyan]")
                else:
                    console.print(f"[red]❌ Unsupported version:[/red] {new_version}")
                    console.print(f"[dim]Supported versions: {', '.join(supported_versions)}[/dim]")
                console.print()
                continue
            elif not query:
                continue

            # Show loading indicator for longer queries
            with console.status("[bold green]Searching Bible database...") as status:
                response = ask_llm_query(query, current_version)
            
            if response:
                console.print()  # Add spacing
                render_results(response)
                console.print()  # Add spacing after results
            else:
                console.print("[red]❌ Failed to get response from API[/red]\n")

        except KeyboardInterrupt:
            console.print("\n\n👋 Exiting...")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Unexpected error:[/red] {e}\n")


if __name__ == "__main__":
    run_llm_cli()



# from rich.console import Console
# from rich.table import Table

# console = Console()
# table = Table(show_header=True, header_style="bold green")
# table.add_column("Reference", style="cyan")
# table.add_column("Text", overflow="fold")  # wraps instead of cutting off

# # Add row (example)
# table.add_row("Isaiah 40:31", "But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles; "
#                               "they shall run, and not be weary; and they shall walk, and not faint.")

# console.print(table)










# # cli.py
# import requests
# import os
# from rich.console import Console
# from rich.table import Table
# from rich.panel import Panel

# BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/bible")
# console = Console()


# def ask_llm_query(query: str):
#     payload = {"query": query, "include_context": True}
#     try:
#         response = requests.post(f"{BASE_URL}/search", json=payload)
#         response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         console.print(f"[red]❌ Request failed:[/red] {e}")
#         return None


# def render_results(response: dict):
#     # CASE: Backend classified it as invalid
#     if response.get("query_type") == "invalid":
#         console.print("[bold red]⚠ This does not appear to be a Bible-related query.[/bold red]")
#         return

#     # SHOW: Query Type
#     match_type = response.get("query_type", "").upper()
#     if match_type == "EXACT_LOOKUP":
#         console.print("[bold green]📌 Match Type: Exact[/bold green]")
#     elif match_type == "SEMANTIC_SEARCH":
#         console.print("[bold yellow]📌 Match Type: Semantic[/bold yellow]")
#     else:
#         console.print("[bold red]📌 Match Type: Unknown[/bold red]")

#     # SHOW: Query
#     console.print(f"\n[blue]🔍 Query:[/blue] {response.get('query', 'N/A')}\n")

#     # AI Response Panel
#     ai_resp = response.get("ai_response", "")
#     if ai_resp:
#         console.rule("[bold blue]🧠 AI Summary")
#         console.print(Panel.fit(ai_resp, border_style="cyan"))
#     else:
#         console.print("[dim]No AI-generated summary provided.[/dim]")

#     # Bible Verses
#     verses = response.get("results", [])
#     if verses:
#         table = Table(title="📖 Bible Verses", show_lines=True)
#         table.add_column("Reference", style="cyan", no_wrap=True)
#         table.add_column("Text", style="white")

#         for verse in verses:
#             table.add_row(verse["reference"], verse["text"])

#         console.print(table)
#     else:
#         console.print("[yellow]⚠️ No verses found.[/yellow]")

#     # Suggestions
#     if response.get("suggestions"):
#         console.print("\n[blue]💡 Suggestions:[/blue]")
#         for s in response["suggestions"]:
#             console.print(f" - {s}")


# def run_llm_cli():
#     console.print("[bold magenta]🧬 Bible RAG LLM Assistant[/bold magenta]")
#     console.print("Type a Bible verse reference or paraphrased quote (e.g., 'John 3:16' or 'God so loved...')\n")
#     console.print("[grey](type 'exit' to quit)[/grey]\n")

#     while True:
#         try:
#             query = console.input("[bold yellow]🔍 Query[/bold yellow]: ").strip()
#             if query.lower() in ["exit", "quit"]:
#                 console.print("\n👋 Goodbye.")
#                 break

#             response = ask_llm_query(query)
#             if response:
#                 render_results(response)

#         except KeyboardInterrupt:
#             console.print("\n👋 Exiting.")
#             break


# if __name__ == "__main__":
#     run_llm_cli()
















# # # cli.py
# # import requests
# # import os
# # from rich.console import Console
# # from rich.table import Table
# # from rich.panel import Panel

# # BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/bible")
# # console = Console()


# # def ask_llm_query(query: str):
# #     payload = {"query": query, "include_context": True}
# #     try:
# #         response = requests.post(f"{BASE_URL}/search", json=payload)
# #         response.raise_for_status()
# #         return response.json()
# #     except Exception as e:
# #         console.print(f"[red]❌ Request failed:[/red] {e}")
# #         return None


# # def render_results(response: dict):
# #     console.rule("[bold blue]🧠 LLM Response")
# #     console.print(Panel.fit(response.get("ai_response", "No AI response available.")))

# #     if response["results"]:
# #         table = Table(title="📖 Bible Verses")
# #         table.add_column("Reference", style="cyan", no_wrap=True)
# #         table.add_column("Text", style="white")

# #         for verse in response["results"]:
# #             table.add_row(verse["reference"], verse["text"])

# #         console.print(table)
# #     else:
# #         console.print("[yellow]⚠️ No verses found.[/yellow]")

# #     if response.get("classification"):
# #         console.print("\n[green]🔎 Classification:[/green]", response["classification"])

# #     if response.get("suggestions"):
# #         console.print("\n[blue]💡 Suggestions:[/blue]")
# #         for s in response["suggestions"]:
# #             console.print(f" - {s}")


# # def run_llm_cli():
# #     console.print("[bold magenta]🧬 Bible RAG LLM Assistant[/bold magenta]")
# #     console.print("Type a Bible verse reference or paraphrased quote (e.g., 'John 3:16' or 'God so loved...')\n")
# #     console.print("[grey](type 'exit' to quit)[/grey]\n")

# #     while True:
# #         try:
# #             query = console.input("[bold yellow]🔍 Query[/bold yellow]: ").strip()
# #             if query.lower() in ["exit", "quit"]:
# #                 console.print("\n👋 Goodbye.")
# #                 break

# #             response = ask_llm_query(query)
# #             if response:
# #                 render_results(response)

# #         except KeyboardInterrupt:
# #             console.print("\n👋 Exiting.")
# #             break


# # if __name__ == "__main__":
# #     run_llm_cli()




