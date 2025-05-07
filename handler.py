from demo_api import run_inference  # твоя функция генерации изображений

def handler(event):
    prompt = event.get('input', {}).get('prompt', 'A cat in space')
    return run_inference(prompt)

