"""
ComfyUI API Client
Simple integration to run ComfyUI workflows from Python.

Prerequisites:
1. ComfyUI must be running on http://127.0.0.1:8188
2. Export your workflow as "API Format" JSON from ComfyUI
"""
import requests
import json
import uuid
import time
import io
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any
import websocket


class ComfyUIClient:
    """Client for interacting with ComfyUI's REST API."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())
        
    def is_available(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        Queue a workflow for execution.
        
        Args:
            workflow: The workflow JSON (API format)
            
        Returns:
            prompt_id for tracking
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        response = requests.post(
            f"{self.base_url}/prompt",
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to queue prompt: {response.text}")
        
        return response.json()["prompt_id"]
    
    def get_history(self, prompt_id: str) -> Optional[Dict]:
        """Get the execution history for a prompt."""
        response = requests.get(f"{self.base_url}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            return history.get(prompt_id)
        return None
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        """Download a generated image from ComfyUI."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        response = requests.get(f"{self.base_url}/view", params=params)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        raise Exception(f"Failed to get image: {response.text}")
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict:
        """
        Wait for a prompt to complete execution.
        
        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum seconds to wait
            
        Returns:
            The execution history
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            
            if history is not None:
                # Check if completed
                if "outputs" in history:
                    return history
            
            time.sleep(0.5)
        
        raise TimeoutError(f"Workflow did not complete within {timeout} seconds")
    
    def run_workflow(
        self,
        workflow: Dict[str, Any],
        timeout: int = 300
    ) -> list[Image.Image]:
        """
        Run a workflow and return the generated images.
        
        Args:
            workflow: The workflow JSON (API format)
            timeout: Maximum seconds to wait
            
        Returns:
            List of generated PIL Images
        """
        # Queue the prompt
        prompt_id = self.queue_prompt(workflow)
        print(f"   📤 Queued workflow: {prompt_id[:8]}...")
        
        # Wait for completion
        history = self.wait_for_completion(prompt_id, timeout)
        
        # Extract images from outputs
        images = []
        for node_id, node_output in history.get("outputs", {}).items():
            if "images" in node_output:
                for img_data in node_output["images"]:
                    image = self.get_image(
                        filename=img_data["filename"],
                        subfolder=img_data.get("subfolder", ""),
                        folder_type=img_data.get("type", "output")
                    )
                    images.append(image)
        
        return images
    
    def generate_image(
        self,
        prompt: str,
        workflow_path: Optional[str] = None,
        workflow: Optional[Dict] = None,
        prompt_node_id: str = "6",
        negative_prompt: str = "blurry, low quality, distorted",
        negative_node_id: str = "7",
        seed: Optional[int] = None,
        seed_node_id: str = "3",
    ) -> Image.Image:
        """
        Generate an image using a workflow template.
        
        Args:
            prompt: The positive prompt text
            workflow_path: Path to workflow JSON file (API format)
            workflow: Or provide workflow dict directly
            prompt_node_id: Node ID for positive prompt (usually "6" or similar)
            negative_prompt: Negative prompt text
            negative_node_id: Node ID for negative prompt
            seed: Random seed (None for random)
            seed_node_id: Node ID for KSampler
            
        Returns:
            Generated PIL Image
        """
        # Load workflow
        if workflow is None:
            if workflow_path is None:
                # Use default workflow
                workflow = self._get_default_workflow()
            else:
                with open(workflow_path, "r") as f:
                    workflow = json.load(f)
        
        # Deep copy to avoid modifying original
        workflow = json.loads(json.dumps(workflow))
        
        # Inject prompt
        if prompt_node_id in workflow:
            workflow[prompt_node_id]["inputs"]["text"] = prompt
        
        # Inject negative prompt
        if negative_node_id in workflow and negative_node_id in workflow:
            workflow[negative_node_id]["inputs"]["text"] = negative_prompt
        
        # Inject seed - ALWAYS randomize if not specified to avoid duplicate images
        if seed_node_id in workflow:
            import random
            actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
            workflow[seed_node_id]["inputs"]["seed"] = actual_seed
        
        # Run workflow
        images = self.run_workflow(workflow)
        
        if not images:
            raise Exception("No images generated")
        
        return images[0]
    
    def _get_default_workflow(self) -> Dict:
        """
        Get a default SDXL workflow.
        This is a basic text-to-image workflow.
        """
        return {
            "3": {
                "inputs": {
                    "seed": 42,
                    "steps": 4,
                    "cfg": 2.0,
                    "sampler_name": "dpmpp_sde",
                    "scheduler": "karras",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": 1024,
                    "height": 576,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": "a beautiful scene",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "blurry, low quality",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "AutoKatha",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }


# Convenience function
def generate_with_comfyui(
    prompt: str,
    workflow_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8188
) -> Image.Image:
    """
    Quick function to generate an image with ComfyUI.
    
    Args:
        prompt: The image prompt
        workflow_path: Optional path to custom workflow JSON
        host: ComfyUI host
        port: ComfyUI port
        
    Returns:
        Generated PIL Image
    """
    client = ComfyUIClient(host, port)
    
    if not client.is_available():
        raise ConnectionError(
            "ComfyUI is not running. Please start ComfyUI first:\n"
            "  cd /path/to/ComfyUI && python main.py"
        )
    
    return client.generate_image(prompt, workflow_path=workflow_path)
