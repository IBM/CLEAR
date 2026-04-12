#!/usr/bin/env python3
"""
Simple launcher script for the Agentic Workflow Dashboard (NiceGUI)
"""

import argparse
import signal
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch the Agentic Workflow Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the dashboard on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Ensure the server shuts down cleanly on termination signals,
    # so the port is released immediately instead of staying in TIME_WAIT.
    def _handle_signal(signum, frame):
        print("\n\nDashboard stopped.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print("Launching Agentic Workflow Dashboard...")
    print("")
    print(f"Dashboard will be available at http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server.")
    print("")

    # Import and run via NiceGUI
    try:
        from nicegui import ui

        # Import the dashboard module to register the page
        script_dir = Path(__file__).parent
        dashboard_path = script_dir / "agentic_workflow_dashboard.py"

        if not dashboard_path.exists():
            print(f"Error: Dashboard file not found at {dashboard_path}")
            sys.exit(1)

        # Import the dashboard module (registers the @ui.page routes)
        import importlib.util
        spec = importlib.util.spec_from_file_location("agentic_workflow_dashboard", dashboard_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Register a shutdown handler to ensure the server cleans up properly
        from nicegui import app
        app.on_shutdown(lambda: print("Server shutting down, releasing port..."))

        ui.run(
            title="Agentic Workflow Dashboard",
            favicon="\U0001F916",
            dark=False,
            port=args.port,
            host=args.host,
            reload=False,
            show=not args.no_open,
            # Increase timeouts for large file uploads
            reconnect_timeout=300.0,  # 5 minutes to reconnect
        )
    except ImportError:
        print("Error: NiceGUI is not installed.")
        print("Please install it with: pip install nicegui")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
