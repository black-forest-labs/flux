from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_handler import RunPodHandler
from demo_api import run_inference  # предполагаем, что ты используешь эту функцию

def handler(event):
    prompt = event.get('input', {}).get('prompt', 'A cat in space')
    return run_inference(prompt)
