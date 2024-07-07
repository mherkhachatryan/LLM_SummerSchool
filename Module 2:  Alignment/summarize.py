from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset
import os
import torch
import evaluate
from transformers import BitsAndBytesConfig

load_dotenv(find_dotenv())

# model_name = 'meta-llama/Meta-Llama-3-8B'
model_name = 'google/gemma-2-9b'
# model_name = 'google/gemma-2-27b'
# model_name = 'meta-llama/Meta-Llama-3-70B'

dtype = torch.bfloat16

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the xsum dataset
dataset = load_dataset("xsum", trust_remote_code=True)

rouge = evaluate.load('rouge')

number_of_samples = 32
validation_random_samples = dataset["validation"].shuffle(seed=42).select(range(number_of_samples))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda',
    token="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",  # TODO do not commit
    torch_dtype=dtype,
    # quantization_config=nf4_config
)
model.eval()
# document = '''Internet searches from the week before the crash were found on the tablet computer used by Andreas Lubitz, Meanwhile, the second "black box" flight recorder from the plane has been recovered. There were no survivors among the 150 people on board the A320 on 24 March. The German prosecutors said internet searches made on the tablet found in Lubitz's Duesseldorf flat included "ways to commit suicide" and "cockpit doors and their security provisions". Spokesman Ralf Herrenbrueck said: "He concerned himself on one hand with medical treatment methods, on the other hand with types and ways of going about a suicide. "In addition, on at least one day he concerned himself with search terms about cockpit doors and their security precautions.'' Prosecutors did not disclose the individual search terms in the browser history but said personal correspondence supported the conclusion Lubitz used the device in the period from 16 to 23 March. Lubitz, 27, had been deemed fit to fly by his employers at Germanwings, a subsidiary of Lufthansa. The first "black box", the voice recorder, was recovered almost immediately at the crash site. Based on that evidence, investigators said they believed Lubitz intentionally crashed Flight 9525, which was travelling from Barcelona to Duesseldorf, taking control of the aircraft while the pilot was locked out of the cockpit. The second "black box" recovered is the flight data recorder (FDR) which should hold technical information on the time of radio transmissions and the plane's acceleration, airspeed, altitude and direction, plus the use of auto-pilot. At a press conference, Marseille prosecutor Brice Robin said there was "reasonable hope" the recorder which was being sent to Paris for examination, would provide useful information. The "completely blackened" equipment was found near a ravine and was not discovered immediately because it was the same colour as the rocks, he said. He said: "The second black box is an indispensable addition to understand what happened especially in the final moment of the flight." He told the media 150 separate DNA profiles had been isolated from the crash site but he stressed that did not mean all the victims had been identified. As each DNA set is matched to a victim, families will be notified immediately, he said, He added 40 mobile phones had been recovered. He said they would be analysed in a laboratory but were "heavily damaged". Also on Thursday, Germanwings said it was unaware that Lubitz had experienced depression while he was training to be a pilot. Lufthansa confirmed on Tuesday that it knew six years ago that the co-pilot had suffered from an episode of "severe depression'' before he finished his flight training. ``We didn't know this,'' said Vanessa Torres, a spokeswoman for Lufthansa subsidiary Germanwings, which hired Lubitz in September 2013. She could not explain why Germanwings had not been informed. The final minutes Lubitz began the jet's descent at 10:31 (09:31 GMT) on 24 March, shortly after the A320 had made its final contact with air traffic control. Little more than eight minutes later, it had crashed into a mountain near Seyne-les-Alpes. What happened in the last 30 minutes of Flight 4U 9525? Who was Andreas Lubitz?'''
# do some tricks with the document to get the `inputs` before passing it to the tokenizer
references = []
generated_texts = []
batch_size = 1
tldr_symbol = "\nTL;DR:"
for i in range(0, len(validation_random_samples), batch_size):
    batch = validation_random_samples[i:i + batch_size]
    input_texts = [document + tldr_symbol for document in batch["document"]]
    references.extend(batch["summary"])
    inputs = tokenizer(
        input_texts,
        padding=True,
        return_tensors='pt',
    ).to("cuda")
    inputs.to(dtype)

    with torch.autocast(device_type="cuda", dtype=dtype):
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 )

    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        if "TL;DR:" in generated_text:
            generated_text = generated_text.split(tldr_symbol)[1].strip()
        generated_texts.append(generated_text)

results = rouge.compute(predictions=generated_texts, references=references)
for reference, generated_text in zip(references, generated_texts):
    print(f"Reference: \n{reference}")
    print("*" * 10)
    print(f"Generated: \n{generated_text}")
    print("----" * 10)

print(results)
