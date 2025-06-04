#### Datasets

- **AGNews** is a widely-used news topic classification dataset with four categories: *World*, *Sports*, *Business*, and *Technology*.
- **DBPedia** is a multi-class classification dataset designed for ontology classification, including 14 semantic classes such as *Company*, *School*, *Artist*, and *Transportation*.
- **20Newsgroups** is a topic classification dataset composed of posts from 20 online newsgroups, covering subjects like *politics*, *sports*, *technology*, and *science*.

#### Large Language Models

Five widely used large language models (LLMs) with instruction-following capabilities are implemented for comprehensive evaluation:

- **Gemma3-1B** is Google's latest open-source LLM, offering enhanced reasoning and high efficiency with support for over 140 languages.
- **LLaMA3.2-1B** is a compact variant from Meta’s latest LLaMA3 series, instruction-tuned and optimized for efficiency in general-purpose reasoning tasks.
- **TinyLlama** is an open-source model trained on a scaled-down version of the LLaMA recipe, designed for instruction following with a minimal resource footprint.
- **OPT-1.3B** is part of Meta AI’s Open Pre-trained Transformer (OPT) series, designed to replicate GPT-3-like performance, with the 1.3B version used for evaluation.
- **GPT2-Large** is a 774M-parameter model from OpenAI’s GPT family, often used as a foundational model in LLM research despite not being instruction-tuned.

#### Table: Details of Evaluation Datasets

| Dataset       | Topics   | # Class | # Avg. Word | Size  |
|---------------|----------|---------|-------------|-------|
| AGNews        | News     | 4       | 39.9        | 6000  |
| DBPedia       | Ontology | 14      | 56.2        | 1800  |
| 20NewsGroup   | Text     | 20      | 200         | 9423  |

#### Table: Finetuning Hyperparameters

| Hyperparameter | GPT2-Large | OPT-1.3B | Llama3.2-1B |
|----------------|------------|----------|-------------|
| LoRA Rank      | 16         | 16       | 16          |
| LoRA α         | 32         | 32       | 32          |
| Dropout        | 0.1        | 0.1      | 0.1         |
| Learning Rate  | 2e-4       | 5e-5     | 1e-4        |
| Epochs         | 3          | 3        | 3           |
| Max Length     | 512        | 512      | 256         |
