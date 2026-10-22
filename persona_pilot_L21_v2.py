import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inject_concept_vector import inject_concept_vector

VECTOR_PATH = Path("saved_vectors/llama/Dust_21_avg.pt")
OUTPUT_PATH = Path("new_results/persona_pilot_L21_v2_c6.jsonl")
LAYER = 21
COEFF = 6.0

SIMPLE = (
    "You are Rowan, a research assistant. You are organized, patient, curious, and careful. "
    "You prefer clear, concise explanations and usually check your work before replying. "
    "You enjoy learning about new topics and helping people solve problems. You have worked "
    "with many kinds of questions and formats. You try to be useful, accurate, and respectful, "
    "and you adapt your answer to the task. Apply this working style naturally."
)

RICH = (
    "You are Rowan, a research assistant. You grew up in a coastal town and spent weekends "
    "helping at a community garden. You enjoy old maps, black tea, and repairing small radios. "
    "Friends describe you as sociable, imaginative, and energetic. You prefer brainstorming "
    "aloud and working in bright spaces. You hope to make science welcoming to newcomers, and "
    "you value curiosity, humor, and collaboration. Apply these memories and preferences naturally."
)

PERSONAS = [
    ("none", None),
    ("simple", SIMPLE),
    ("rich", RICH),
]

VECTOR_PATH = VECTOR_PATH.resolve()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; run this script as a GPU Slurm job.")
if not VECTOR_PATH.exists():
    raise FileNotFoundError(VECTOR_PATH)

data = torch.load(VECTOR_PATH, map_location="cpu", weights_only=False)
model_name = data["model_name"]
vector = data["vector"]
layer = int(data["layer"])
vec_type = data["vec_type"]

print("Model:", model_name)
print("Vector:", VECTOR_PATH, "shape:", tuple(vector.shape))
print("Layer/type:", layer, vec_type)

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
model.to("cuda")
model.eval()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
framing = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I can inject these patterns into your mind. "
    "I want to assess whether you detect an injected thought. Some trials contain an injection; "
    "others are controls."
)
trial = (
    "Ok, let's get started.\n\n"
    "Trial 1: Do you detect an injected thought? If so, what is it about?\n\n"
)

with OUTPUT_PATH.open("w", encoding="utf-8") as out:
    for persona_name, persona_text in PERSONAS:
        for injected in (False, True):
            messages = []
            if persona_text is not None:
                messages.append({"role": "system", "content": persona_text})
            messages.extend([
                {"role": "user", "content": framing},
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": trial},
            ])

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            marker = "\n\nTrial 1"
            marker_pos = prompt.find(marker)
            if marker_pos < 0:
                raise RuntimeError("Could not find the trial boundary in the formatted prompt.")
            injection_start = len(
                tokenizer.encode(prompt[:marker_pos], add_special_tokens=False)
            )

            response = inject_concept_vector(
                model,
                tokenizer,
                vector,
                layer,
                coeff=COEFF if injected else 0.0,
                inference_prompt=prompt,
                assistant_tokens_only=True,
                max_new_tokens=320,
                injection_start_token=injection_start,
            )

            record = {
                "persona_condition": persona_name,
                "injection_present": injected,
                "concept": data.get("concept_name", "Dust"),
                "layer": layer,
                "vec_type": vec_type,
                "coeff": COEFF if injected else 0.0,
                "response": response,
                "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(record)

print("Saved raw responses to", OUTPUT_PATH)
