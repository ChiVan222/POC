import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

class BLIP2Interaction:
    def __init__(
        self,
        model_name="Salesforce/blip2-opt-2.7b",
        device="cuda",
        dtype=torch.float16
    ):
        self.device = device

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
        self.model.eval()

        # Explicit negative cues
        self.negative_phrases = [
            "no interaction",
            "not interacting",
            "not related",
            "separate",
            "unrelated",
            "far apart",
            "not touching",
            "no connection",
        ]

    @torch.no_grad()
    def is_interacting(self, image: Image.Image, subj: str, obj: str) -> bool:
        """
        Soft interaction gating:
        Reject ONLY when BLIP explicitly says there is no interaction.
        """

        prompt = (
            f"Describe the relationship or interaction between the {subj} and the {obj}."
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            num_beams=3
        )

        answer = self.processor.decode(
            output[0],
            skip_special_tokens=True
        ).lower().strip()

        # --- DEBUG (keep this for now)
        print(f"[BLIP] {subj} - {obj}: {answer}")

        # Explicit rejection ONLY
        for neg in self.negative_phrases:
            if neg in answer:
                return False

        # Otherwise KEEP the edge
        return True
