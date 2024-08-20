ijun111:main$ python -m flux.api --prompt="A beautiful beach" url
https:<...>/sample.jpg

# generate and save the result
$ python -m flux.api --prompt="A beautiful beach" save outputs/api

# open the image directly
$ python -m flux.api --prompt="A beautiful beach" image showflux.apirequest.urlfrom flux.api import ImageRequest

# this will create an api request directly but not block until the generation is finished
request = ImageRequest("A beautiful beach")
# or: request = ImageRequest("A beautiful beach", api_key="your_key_here")

# any of the following will block until the generation is finished
request.url
# -> https:<...>/sample.jpg
request.bytes
# -> b"..." bytes for the generated image
request.save("outputs/api.jpg")
# saves the sample to local storage
request.image
# -> a PIL imagerequest.bytesrequest.imageimport torch
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4, #use a larger number if you are using [dev]
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]
image.save("flux-schnell.png")42https://github.com/huggingface/diffusers.gitpip install git+https://github.com/huggingface/diffusers.git# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.
