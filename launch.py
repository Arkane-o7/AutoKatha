#!/usr/bin/env python3
"""
AutoKatha Unified Launcher
Single command to start ComfyUI + Ollama + AutoKatha web interface

Usage:
    python launch.py              # Start everything
    python launch.py --no-comfyui # Skip ComfyUI
    python launch.py --no-ollama  # Skip Ollama
    python launch.py --check      # Just check services status
"""
import subprocess
import sys
import time
import argparse
import requests
import threading
import signal
import os
from pathlib import Path

# ANSI colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_banner():
    """Print AutoKatha banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ██╗   ██╗████████╗ ██████╗ ██╗  ██╗ █████╗     ║
    ║    ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██║ ██╔╝██╔══██╗    ║
    ║    ███████║██║   ██║   ██║   ██║   ██║█████╔╝ ███████║    ║
    ║    ██╔══██║██║   ██║   ██║   ██║   ██║██╔═██╗ ██╔══██║    ║
    ║    ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██╗██║  ██║    ║
    ║    ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ║
    ║                                                           ║
    ║           Text & Comics → Animated Videos                 ║
    ╚═══════════════════════════════════════════════════════════╝
{Colors.END}"""
    print(banner)


class ServiceManager:
    """Manages external services for AutoKatha."""
    
    def __init__(self):
        self.processes = {}
        self.running = True
        
        # Service configurations
        self.services = {
            'ollama': {
                'name': 'Ollama LLM Server',
                'check_url': 'http://localhost:11434/api/tags',
                'start_cmd': self._get_ollama_cmd(),
                'required': True,
                'port': 11434,
            },
            'comfyui': {
                'name': 'ComfyUI Image Generation',
                'check_url': 'http://127.0.0.1:8000/system_stats',
                'start_cmd': self._get_comfyui_cmd(),
                'required': False,
                'port': 8000,
            },
        }
    
    def _get_ollama_cmd(self):
        """Get Ollama start command based on OS."""
        if sys.platform == 'win32':
            return ['ollama', 'serve']
        elif sys.platform == 'darwin':
            return ['ollama', 'serve']
        else:
            return ['ollama', 'serve']
    
    def _get_comfyui_cmd(self):
        """Get ComfyUI start command - returns None if not configured."""
        # Check common ComfyUI locations
        possible_paths = [
            Path.home() / 'ComfyUI',
            Path.home() / 'Documents' / 'ComfyUI',
            Path('C:/ComfyUI'),
            Path('D:/ComfyUI'),
            Path('/opt/ComfyUI'),
        ]
        
        # Check environment variable
        comfyui_path = os.environ.get('COMFYUI_PATH')
        if comfyui_path:
            possible_paths.insert(0, Path(comfyui_path))
        
        for path in possible_paths:
            main_py = path / 'main.py'
            if main_py.exists():
                return [sys.executable, str(main_py), '--port', '8000']
        
        return None
    
    def check_service(self, service_id: str, timeout: float = 2.0) -> bool:
        """Check if a service is running."""
        service = self.services.get(service_id)
        if not service:
            return False
        
        try:
            response = requests.get(service['check_url'], timeout=timeout)
            return response.status_code == 200
        except:
            return False
    
    def start_service(self, service_id: str) -> bool:
        """Start a service if not running."""
        service = self.services.get(service_id)
        if not service:
            print(f"{Colors.RED}✗ Unknown service: {service_id}{Colors.END}")
            return False
        
        name = service['name']
        
        # Check if already running
        if self.check_service(service_id):
            print(f"{Colors.GREEN}✓ {name} is already running{Colors.END}")
            return True
        
        # Check if we have a start command
        cmd = service['start_cmd']
        if not cmd:
            if service['required']:
                print(f"{Colors.YELLOW}⚠ {name} not found. Please start manually.{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠ {name} not configured (optional){Colors.END}")
            return not service['required']
        
        # Start the service
        print(f"{Colors.BLUE}→ Starting {name}...{Colors.END}")
        
        try:
            # Start process in background
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid
                )
            
            self.processes[service_id] = process
            
            # Wait for service to be ready
            for i in range(30):  # 30 second timeout
                if self.check_service(service_id, timeout=1.0):
                    print(f"{Colors.GREEN}✓ {name} started successfully{Colors.END}")
                    return True
                time.sleep(1)
                print(f"  Waiting for {name}... ({i+1}s)")
            
            print(f"{Colors.RED}✗ {name} failed to start (timeout){Colors.END}")
            return False
            
        except FileNotFoundError:
            print(f"{Colors.RED}✗ {name} command not found: {cmd[0]}{Colors.END}")
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to start {name}: {e}{Colors.END}")
            return False
    
    def stop_service(self, service_id: str):
        """Stop a service."""
        if service_id in self.processes:
            process = self.processes[service_id]
            try:
                if sys.platform == 'win32':
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                print(f"{Colors.YELLOW}→ Stopped {self.services[service_id]['name']}{Colors.END}")
            except:
                pass
            del self.processes[service_id]
    
    def stop_all(self):
        """Stop all managed services."""
        for service_id in list(self.processes.keys()):
            self.stop_service(service_id)
    
    def check_ollama_model(self, model: str = "gemma3:4b") -> bool:
        """Check if required Ollama model is available."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                # Check for exact match or base name match
                base_model = model.split(':')[0]
                for m in model_names:
                    if model in m or base_model in m:
                        return True
                
                return False
        except:
            return False
    
    def pull_ollama_model(self, model: str = "gemma3:4b"):
        """Pull an Ollama model if not present."""
        if self.check_ollama_model(model):
            print(f"{Colors.GREEN}✓ Ollama model '{model}' is available{Colors.END}")
            return True
        
        print(f"{Colors.BLUE}→ Downloading Ollama model '{model}'...{Colors.END}")
        print(f"  (This may take a few minutes on first run)")
        
        try:
            result = subprocess.run(
                ['ollama', 'pull', model],
                capture_output=False,
                timeout=600  # 10 minute timeout
            )
            return result.returncode == 0
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to pull model: {e}{Colors.END}")
            return False
    
    def status_report(self):
        """Print status of all services."""
        print(f"\n{Colors.BOLD}Service Status:{Colors.END}")
        print("─" * 40)
        
        for service_id, service in self.services.items():
            status = self.check_service(service_id)
            status_icon = f"{Colors.GREEN}●{Colors.END}" if status else f"{Colors.RED}○{Colors.END}"
            status_text = "Running" if status else "Stopped"
            required = "(required)" if service['required'] else "(optional)"
            
            print(f"  {status_icon} {service['name']}: {status_text} {Colors.YELLOW}{required}{Colors.END}")
            print(f"      Port: {service['port']}")
        
        # Check Ollama model
        if self.check_service('ollama'):
            model_status = self.check_ollama_model()
            model_icon = f"{Colors.GREEN}●{Colors.END}" if model_status else f"{Colors.YELLOW}○{Colors.END}"
            model_text = "Available" if model_status else "Not downloaded"
            print(f"  {model_icon} Ollama Model (gemma3:4b): {model_text}")
        
        print("─" * 40)


def check_dependencies():
    """Check for required Python packages."""
    missing = []
    
    required = ['gradio', 'torch', 'requests', 'PIL', 'moviepy']
    optional = ['TTS', 'edge_tts', 'whisper']
    
    for pkg in required:
        try:
            if pkg == 'PIL':
                __import__('PIL')
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"{Colors.RED}✗ Missing required packages: {', '.join(missing)}{Colors.END}")
        print(f"  Run: pip install -r requirements.txt")
        return False
    
    print(f"{Colors.GREEN}✓ All required packages installed{Colors.END}")
    return True


def launch_autokatha(port: int = 7861):
    """Launch the AutoKatha web interface."""
    print(f"\n{Colors.BOLD}Launching AutoKatha Web Interface...{Colors.END}")
    print(f"  URL: http://127.0.0.1:{port}")
    print(f"  Press Ctrl+C to stop\n")
    
    # Import and launch
    from app_text import create_interface
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True
    )


def main():
    parser = argparse.ArgumentParser(
        description='AutoKatha Unified Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py              Start all services and AutoKatha
  python launch.py --check      Check status of all services
  python launch.py --no-comfyui Start without ComfyUI
  python launch.py --port 7860  Use different port for web UI
        """
    )
    
    parser.add_argument('--check', action='store_true',
                        help='Only check service status, don\'t start anything')
    parser.add_argument('--no-ollama', action='store_true',
                        help='Skip starting Ollama (use if already running)')
    parser.add_argument('--no-comfyui', action='store_true',
                        help='Skip starting ComfyUI (will use diffusers fallback)')
    parser.add_argument('--port', type=int, default=7861,
                        help='Port for AutoKatha web interface (default: 7861)')
    parser.add_argument('--pull-model', type=str, default=None,
                        help='Pull a specific Ollama model before starting')
    
    args = parser.parse_args()
    
    print_banner()
    
    manager = ServiceManager()
    
    # Just check status
    if args.check:
        manager.status_report()
        return
    
    # Check Python dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print(f"\n{Colors.BOLD}Starting Services...{Colors.END}")
    print("─" * 40)
    
    try:
        # Start Ollama
        if not args.no_ollama:
            if not manager.start_service('ollama'):
                print(f"{Colors.RED}✗ Ollama is required but failed to start{Colors.END}")
                sys.exit(1)
            
            # Pull model if needed
            model = args.pull_model or "gemma3:4b"
            if not manager.pull_ollama_model(model):
                print(f"{Colors.YELLOW}⚠ Model pull failed, but continuing...{Colors.END}")
        
        # Start ComfyUI (optional)
        if not args.no_comfyui:
            manager.start_service('comfyui')
        
        # Print final status
        manager.status_report()
        
        # Launch AutoKatha
        launch_autokatha(port=args.port)
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Shutting down...{Colors.END}")
    finally:
        # Cleanup
        manager.stop_all()
        print(f"{Colors.GREEN}✓ AutoKatha stopped{Colors.END}")


if __name__ == "__main__":
    main()
