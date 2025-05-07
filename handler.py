import runpod
from demo_api import run_inference  # Убедись, что demo_api.py существует и содержит функцию run_inference

def handler(job):
    prompt = job.get('input', {}).get('prompt', 'A cat in space')
    return run_inference(prompt)

runpod.serverless.start({"handler": handler})
